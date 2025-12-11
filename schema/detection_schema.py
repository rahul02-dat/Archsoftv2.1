from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

'''REQUESTS'''

class DetectionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    location: Optional[str] = Field(None, description="Optional location information")

'''RESPONSES'''

class MatchResponse(BaseModel):
    is_match: bool
    person_id: Optional[str] = None
    confidence_score: Optional[float] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_detections: Optional[int] = None
    message: str

class DetectionResponse(BaseModel):
    success: bool
    is_match: bool
    person_id: str
    confidence_score: float
    first_seen: datetime
    last_seen: datetime
    total_detections: int
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