package dto


type PredictionEventRequest struct {
	Gesture     string  `json:"gesture"`
	Confidence  float64 `json:"confidence"`
	DeviceID    string  `json:"device_id"`
	ModelVersion string `json:"model_version"`
	Timestamp   string  `json:"timestamp"`
}