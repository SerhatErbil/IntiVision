package entities

import "time"

type PredictionEvent struct {
	ID           int64     `json:"id"`
	Gesture      string    `json:"gesture"`
	Confidence   float64   `json:"confidence"`
	DeviceID     string    `json:"device_id"`
	ModelVersion string    `json:"model_version"`
	PredictedAt  time.Time `json:"predicted_at"`
	CreatedAt    time.Time `json:"created_at"`
}
