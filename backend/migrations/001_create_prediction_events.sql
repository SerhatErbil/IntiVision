CREATE TABLE prediction_events (
    id BIGSERIAL PRIMARY KEY,
    gesture TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    device_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);