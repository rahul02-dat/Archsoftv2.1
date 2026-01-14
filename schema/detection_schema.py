from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class DetectionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    location: Optional[str] = Field(default=None, description="Optional location information")
    increment_detection: bool = Field(default=True, description="Whether to increment detection count")

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class SingleFaceDetection(BaseModel):
    is_match: bool
    person_id: str
    confidence_score: float
    first_seen: datetime
    last_seen: datetime
    total_detections: int
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    success: bool
    faces_detected: int
    detections: List[SingleFaceDetection]
    message: str

class PersonListResponse(BaseModel):
    id: int
    person_id: str
    first_seen: datetime
    last_seen: datetime
    total_detections: int
    average_confidence: float

    class Config:
        from_attributes = True