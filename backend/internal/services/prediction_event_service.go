package services

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/SerhatErbil/IntiVision/backend/internal/dto"
	"github.com/SerhatErbil/IntiVision/backend/internal/entities"
	"github.com/SerhatErbil/IntiVision/backend/internal/repositories"
)

type ValidationError struct {
	Message string
}

func (e *ValidationError) Error() string {
	return e.Message
}

type PredictionEventService struct {
	repository *repositories.PredictionEventRepository
}

func NewPredictionEventService(
	repository *repositories.PredictionEventRepository,
) *PredictionEventService {
	return &PredictionEventService{
		repository: repository,
	}
}

func (s *PredictionEventService) CreatePredictionEvent(
	ctx context.Context,
	request dto.PredictionEventRequest,
) (*entities.PredictionEvent, error) {
	request.Gesture = strings.TrimSpace(request.Gesture)
	request.DeviceID = strings.TrimSpace(request.DeviceID)
	request.ModelVersion = strings.TrimSpace(request.ModelVersion)
	request.Timestamp = strings.TrimSpace(request.Timestamp)

	if request.Gesture == "" {
		return nil, &ValidationError{
			Message: "gesture is required",
		}
	}

	if request.DeviceID == "" {
		return nil, &ValidationError{
			Message: "device_id is required",
		}
	}

	if request.ModelVersion == "" {
		return nil, &ValidationError{
			Message: "model_version is required",
		}
	}

	if request.Timestamp == "" {
		return nil, &ValidationError{
			Message: "timestamp is required",
		}
	}

	if request.Confidence < 0 || request.Confidence > 1 {
		return nil, &ValidationError{
			Message: "confidence must be between 0 and 1",
		}
	}

	predictedAt, err := time.Parse(time.RFC3339, request.Timestamp)
	if err != nil {
		return nil, &ValidationError{
			Message: "timestamp must be in RFC3339 format",
		}
	}

	event := &entities.PredictionEvent{
		Gesture:      request.Gesture,
		Confidence:   request.Confidence,
		DeviceID:     request.DeviceID,
		ModelVersion: request.ModelVersion,
		PredictedAt:  predictedAt,
	}

	if err := s.repository.Create(ctx, event); err != nil {
		return nil, fmt.Errorf(
			"failed to create prediction event: %w",
			err,
		)
	}

	return event, nil
}

func (s *PredictionEventService) GetPredictionEvents(
	ctx context.Context,
	limit int,
) ([]entities.PredictionEvent, error) {
	if limit <= 0 {
		limit = 20
	}

	if limit > 100 {
		limit = 100
	}

	return s.repository.GetAll(ctx, limit)
}