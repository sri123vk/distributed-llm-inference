FROM golang:1.22 AS build
WORKDIR /src
COPY go.mod ./
COPY scheduler ./scheduler
RUN CGO_ENABLED=0 go build -o /scheduler ./scheduler
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /scheduler /scheduler
EXPOSE 8002
ENTRYPOINT ["/scheduler"]
