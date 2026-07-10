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

func (r *PredictionEventRepository) GetAll(
	ctx context.Context,
	limit int,
) ([]entities.PredictionEvent, error) {
	query := `
		SELECT
			id,
			gesture,
			confidence,
			device_id,
			model_version,
			predicted_at,
			created_at
		FROM prediction_events
		ORDER BY created_at DESC
		LIMIT $1
	`

	rows, err := r.db.Query(ctx, query, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch prediction events: %w", err)
	}
	defer rows.Close()

	events := make([]entities.PredictionEvent, 0)

	for rows.Next() {
		var event entities.PredictionEvent

		if err := rows.Scan(
			&event.ID,
			&event.Gesture,
			&event.Confidence,
			&event.DeviceID,
			&event.ModelVersion,
			&event.PredictedAt,
			&event.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan prediction event: %w", err)
		}

		events = append(events, event)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("failed while reading prediction events: %w", err)
	}

	return events, nil
}
