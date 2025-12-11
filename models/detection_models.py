from sqlalchemy import Column, Integer, String, DateTime, Float
from datetime import datetime
import pytz
from database import Base

IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    return datetime.now(IST)

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String(50), nullable=False, index=True)
    detection_time = Column(DateTime, default=get_ist_time, nullable=False)
    confidence_score = Column(Float, nullable=False)
    location = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=get_ist_time)