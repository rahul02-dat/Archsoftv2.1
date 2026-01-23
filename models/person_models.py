from sqlalchemy import Column, Integer, String, DateTime, LargeBinary, Float
from datetime import datetime
import pytz
from database import Base

IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    return datetime.now(IST)

class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String(50), unique=True, nullable=False, index=True)
    first_seen = Column(DateTime, default=get_ist_time, nullable=False)
    last_seen = Column(DateTime, default=get_ist_time, onupdate=get_ist_time, nullable=False)
    total_detections = Column(Integer, default=1)
    average_confidence = Column(Float, default=0.0)
    embedding = Column(LargeBinary, nullable=False)
    role = Column(String(20), default="Customer", nullable=False)  # New: Employee or Customer
    created_at = Column(DateTime, default=get_ist_time)
    updated_at = Column(DateTime, default=get_ist_time, onupdate=get_ist_time)