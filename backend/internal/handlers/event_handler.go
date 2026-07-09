package handlers

import (
	"fmt"

	"github.com/gofiber/fiber/v2"

	"github.com/SerhatErbil/IntiVision/backend/internal/dto"
)

func HandleCreatePredictionEvent(c *fiber.Ctx) error {
	var request dto.PredictionEventRequest

	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
		})
	}

	fmt.Println("Prediction event received:", request)

	return c.Status(fiber.StatusCreated).JSON(fiber.Map{
		"status": "received",
		"event":  request,
	})
}