package services

import (
	"context"
	"fmt"
	"time"

	"github.com/SerhatErbil/IntiVision/backend/internal/dto"
	"github.com/SerhatErbil/IntiVision/backend/internal/entities"
	"github.com/SerhatErbil/IntiVision/backend/internal/repositories"
)

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

	predictedAt, err := time.Parse(time.RFC3339, request.Timestamp)
	if err != nil {
		return nil, fmt.Errorf("invalid timestamp: %w", err)
	}

	event := &entities.PredictionEvent{
		Gesture:      request.Gesture,
		Confidence:   request.Confidence,
		DeviceID:     request.DeviceID,
		ModelVersion: request.ModelVersion,
		PredictedAt:  predictedAt,
	}

	err = s.repository.Create(ctx, event)
	if err != nil {
		return nil, err
	}

	return event, nil
}