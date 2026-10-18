FROM golang:1.22 AS build
WORKDIR /src
COPY go.mod ./
COPY gateway ./gateway
RUN CGO_ENABLED=0 go build -o /gateway ./gateway
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /gateway /gateway
EXPOSE 8000
ENTRYPOINT ["/gateway"]
