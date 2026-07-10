package handlers

import (
	"github.com/SerhatErbil/IntiVision/backend/internal/dto"
	"github.com/SerhatErbil/IntiVision/backend/internal/services"
	"github.com/gofiber/fiber/v2"
)

type PredictionEventHandler struct {
	service *services.PredictionEventService
}

func NewPredictionEventHandler(
	service *services.PredictionEventService,
) *PredictionEventHandler {
	return &PredictionEventHandler{
		service: service,
	}
}

func (h *PredictionEventHandler) HandleCreatePredictionEvent(c *fiber.Ctx) error {
	var request dto.PredictionEventRequest

	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "invalid request body",
		})
	}

	event, err := h.service.CreatePredictionEvent(c.Context(), request)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return c.Status(fiber.StatusCreated).JSON(fiber.Map{
		"status": "created",
		"event":  event,
	})
}