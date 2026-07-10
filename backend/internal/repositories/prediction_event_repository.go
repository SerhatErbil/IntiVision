package repositories

import (
	"context"
	"fmt"

	"github.com/SerhatErbil/IntiVision/backend/internal/entities"
	"github.com/jackc/pgx/v5/pgxpool"
)

type PredictionEventRepository struct {
	db *pgxpool.Pool
}

func NewPredictionEventRepository(db *pgxpool.Pool) *PredictionEventRepository {
	return &PredictionEventRepository{
		db: db,
	}
}

func (r *PredictionEventRepository) Create(
	ctx context.Context,
	event *entities.PredictionEvent,
) error {
	query := `
		INSERT INTO prediction_events (
			gesture,
			confidence,
			device_id,
			model_version,
			predicted_at
		)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, created_at
	`

	err := r.db.QueryRow(
		ctx,
		query,
		event.Gesture,
		event.Confidence,
		event.DeviceID,
		event.ModelVersion,
		event.PredictedAt,
	).Scan(
		&event.ID,
		&event.CreatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to create prediction event: %w", err)
	}

	return nil
}
