package main

import (
	"log"

	"github.com/SerhatErbil/IntiVision/backend/internal/database"
	"github.com/SerhatErbil/IntiVision/backend/internal/handlers"
	"github.com/SerhatErbil/IntiVision/backend/internal/repositories"
	"github.com/SerhatErbil/IntiVision/backend/internal/services"
	"github.com/gofiber/fiber/v2"
)

func main() {
	db, err := database.ConnectPostgreSQL()
	if err != nil {
		log.Fatal("PostgreSQL connection failed: ", err)
	}
	defer db.Close()

	log.Println("PostgreSQL connected successfully")

	predictionEventRepository :=
		repositories.NewPredictionEventRepository(db)

	predictionEventService :=
		services.NewPredictionEventService(predictionEventRepository)

	predictionEventHandler :=
		handlers.NewPredictionEventHandler(predictionEventService)

	app := fiber.New()

	api := app.Group("/api/v1")

	api.Post(
		"/events",
		predictionEventHandler.HandleCreatePredictionEvent,
	)
	api.Get(
		"/events",
		predictionEventHandler.HandleGetPredictionEvents,
	)

	log.Println("IntiVision backend running on port 8080")

	if err := app.Listen(":8080"); err != nil {
		log.Fatal(err)
	}
}
