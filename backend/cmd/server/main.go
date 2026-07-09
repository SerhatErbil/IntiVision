package main

import (
	"log"

	"github.com/SerhatErbil/IntiVision/backend/internal/handlers"
	"github.com/gofiber/fiber/v2"
)

func main() {
	app := fiber.New()

	api := app.Group("/api/v1")

	api.Post("/events", handlers.HandleCreatePredictionEvent)

	log.Println("IntiVision backend running on port 8080")

	if err := app.Listen(":8080"); err != nil {
		log.Fatal(err)
	}
}