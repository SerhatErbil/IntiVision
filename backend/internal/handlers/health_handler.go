package handlers

import (
	"context"
	"log"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/jackc/pgx/v5/pgxpool"
)

type HealthHandler struct {
	db *pgxpool.Pool
}

func NewHealthHandler(db *pgxpool.Pool) *HealthHandler {
	return &HealthHandler{
		db: db,
	}
}

func (h *HealthHandler) HandleHealth(c *fiber.Ctx) error {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := h.db.Ping(ctx); err != nil {
		log.Printf("health check database ping failed: %v", err)

		return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{
			"status":   "error",
			"database": "disconnected",
			"service":  "intivision-backend",
		})
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{
		"status":   "ok",
		"database": "connected",
		"service":  "intivision-backend",
	})
}
