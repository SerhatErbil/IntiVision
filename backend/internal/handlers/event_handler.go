package handlers

import (
	"errors"
	"log"
	"strconv"

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

func (h *PredictionEventHandler) HandleCreatePredictionEvent(
	c *fiber.Ctx,
) error {
	var request dto.PredictionEventRequest

	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "invalid request body",
		})
	}

	event, err := h.service.CreatePredictionEvent(
		c.Context(),
		request,
	)
	if err != nil {
		var validationError *services.ValidationError

		if errors.As(err, &validationError) {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": validationError.Error(),
			})
		}

		log.Printf(
			"failed to create prediction event: %v",
			err,
		)

		return c.Status(
			fiber.StatusInternalServerError,
		).JSON(fiber.Map{
			"error": "failed to create prediction event",
		})
	}

	return c.Status(fiber.StatusCreated).JSON(fiber.Map{
		"status": "created",
		"event":  event,
	})
}

func (h *PredictionEventHandler) HandleGetPredictionEvents(
	c *fiber.Ctx,
) error {
	limit := 20

	if limitQuery := c.Query("limit"); limitQuery != "" {
		parsedLimit, err := strconv.Atoi(limitQuery)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "limit must be a valid integer",
			})
		}

		limit = parsedLimit
	}

	events, err := h.service.GetPredictionEvents(
		c.Context(),
		limit,
	)
	if err != nil {
		log.Printf(
			"failed to fetch prediction events: %v",
			err,
		)

		return c.Status(
			fiber.StatusInternalServerError,
		).JSON(fiber.Map{
			"error": "failed to fetch prediction events",
		})
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{
		"count":  len(events),
		"events": events,
	})
}
