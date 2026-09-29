from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for health status check endpoint response."""
    status: str = Field(..., description="Overall health status", json_schema_extra={"example": "ok"})
    app_name: str = Field(..., description="Application name")
    environment: str = Field(..., description="Running environment")
    version: str = Field(default="1.0.0", description="API version")
