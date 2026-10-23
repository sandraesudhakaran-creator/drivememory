"""AWS Lambda entry point (API Gateway HTTP API -> FastAPI via Mangum)."""

from mangum import Mangum

from .app import app

handler = Mangum(app, lifespan="off")
