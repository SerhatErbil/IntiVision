FROM golang:1.25-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./

RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build \
    -o /app/bin/intivision-backend \
    ./cmd/server

FROM alpine:3.21

WORKDIR /app

RUN apk add --no-cache ca-certificates

COPY --from=builder \
    /app/bin/intivision-backend \
    /app/intivision-backend

EXPOSE 8080

CMD ["/app/intivision-backend"]