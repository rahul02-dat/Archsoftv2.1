from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from schema import DetectionRequest, DetectionResponse, PersonListResponse
from models import Person, Detection
from utils import face_recognition_system
from datetime import datetime
import pytz

detection_router = APIRouter(prefix="/detection", tags=["Detection"])

IST = pytz.timezone('Asia/Kolkata')

def generate_person_id(db: Session) -> str:
    """Generate unique person ID in format AUTO_P00001"""
    last_person = db.query(Person).order_by(Person.id.desc()).first()
    
    if last_person:
        # Extract number from last person_id (AUTO_P00001 -> 1)
        last_num = int(last_person.person_id.split('_P')[1])
        new_num = last_num + 1
    else:
        new_num = 1
    
    return f"AUTO_P{new_num:05d}"

@detection_router.post("/recognize", response_model=DetectionResponse)
async def recognize_face(
    request: DetectionRequest,
    db: Session = Depends(get_db)
):
    """
    Automated face recognition endpoint
    - If face matches: Returns match info and updates last_seen
    - If face is new: Auto-registers with new person_id
    """
    try:
        # Process the image
        embedding, message = face_recognition_system.process_detection_image(request.image_base64)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail=message)
        
        # Get all registered persons
        persons = db.query(Person).all()
        
        # Check for match
        if persons:
            stored_embeddings = [person.embedding for person in persons]
            is_match, confidence, match_idx = face_recognition_system.verify_face(
                embedding, 
                stored_embeddings
            )
            
            if is_match:
                # Update existing person
                matched_person = persons[match_idx]
                
                # Update last_seen and statistics
                matched_person.last_seen = datetime.now(IST)
                matched_person.total_detections += 1
                
                # Update average confidence
                old_avg = matched_person.average_confidence
                old_count = matched_person.total_detections - 1
                new_avg = ((old_avg * old_count) + confidence) / matched_person.total_detections
                matched_person.average_confidence = new_avg
                
                # Log detection
                detection_record = Detection(
                    person_id=matched_person.person_id,
                    detection_time=datetime.now(IST),
                    confidence_score=confidence,
                    location=request.location or "Unknown"
                )
                
                db.add(detection_record)
                db.commit()
                db.refresh(matched_person)
                
                return DetectionResponse(
                    success=True,
                    is_match=True,
                    person_id=matched_person.person_id,
                    confidence_score=confidence,
                    first_seen=matched_person.first_seen,
                    last_seen=matched_person.last_seen,
                    total_detections=matched_person.total_detections,
                    message=f"MATCH FOUND: {matched_person.person_id}"
                )
        
        # No match found - Auto-register new person
        new_person_id = generate_person_id(db)
        embedding_bytes = face_recognition_system.serialize_embedding(embedding)
        
        new_person = Person(
            person_id=new_person_id,
            first_seen=datetime.now(IST),
            last_seen=datetime.now(IST),
            total_detections=1,
            average_confidence=1.0,  # First detection, perfect match with itself
            embedding=embedding_bytes
        )
        
        db.add(new_person)
        
        # Log first detection
        detection_record = Detection(
            person_id=new_person_id,
            detection_time=datetime.now(IST),
            confidence_score=1.0,
            location=request.location or "Unknown"
        )
        
        db.add(detection_record)
        db.commit()
        db.refresh(new_person)
        
        return DetectionResponse(
            success=True,
            is_match=False,
            person_id=new_person.person_id,
            confidence_score=1.0,
            first_seen=new_person.first_seen,
            last_seen=new_person.last_seen,
            total_detections=1,
            message=f"NEW PERSON REGISTERED: {new_person_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error in recognize_face: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@detection_router.get("/persons")
async def list_all_persons(db: Session = Depends(get_db)):
    """List all detected persons"""
    try:
        persons = db.query(Person).order_by(Person.last_seen.desc()).all()
        
        return {
            "total": len(persons),
            "persons": [
                PersonListResponse(
                    id=person.id,
                    person_id=person.person_id,
                    first_seen=person.first_seen,
                    last_seen=person.last_seen,
                    total_detections=person.total_detections,
                    average_confidence=person.average_confidence
                )
                for person in persons
            ]
        }
    except Exception as e:
        print(f"Error in list_all_persons: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching persons: {str(e)}")

@detection_router.get("/person/{person_id}/history")
async def get_person_history(
    person_id: str,
    db: Session = Depends(get_db)
):
    """Get detection history for a specific person"""
    person = db.query(Person).filter(Person.person_id == person_id).first()
    
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    
    detections = db.query(Detection).filter(
        Detection.person_id == person_id
    ).order_by(Detection.detection_time.desc()).all()
    
    return {
        "person_id": person.person_id,
        "first_seen": person.first_seen,
        "last_seen": person.last_seen,
        "total_detections": person.total_detections,
        "average_confidence": person.average_confidence,
        "detection_history": [
            {
                "id": det.id,
                "detection_time": det.detection_time,
                "confidence_score": det.confidence_score,
                "location": det.location
            }
            for det in detections
        ]
    }

@detection_router.delete("/person/{person_id}")
async def delete_person(person_id: str, db: Session = Depends(get_db)):
    """Delete a person and all their detection records"""
    try:
        person = db.query(Person).filter(Person.person_id == person_id).first()
        
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Delete all detection records
        db.query(Detection).filter(Detection.person_id == person_id).delete()
        
        # Delete person
        db.delete(person)
        db.commit()
        
        return {"message": f"Person {person_id} and all detection records deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error in delete_person: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting person: {str(e)}")

@detection_router.delete("/reset-all")
async def reset_all_data(db: Session = Depends(get_db)):
    """Reset all persons and detections (USE WITH CAUTION)"""
    try:
        # Delete all detections
        db.query(Detection).delete()
        
        # Delete all persons
        db.query(Person).delete()
        
        db.commit()
        
        return {"message": "All data reset successfully"}
    except Exception as e:
        db.rollback()
        print(f"Error in reset_all_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting data: {str(e)}")

@detection_router.get("/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        total_persons = db.query(Person).count()
        total_detections = db.query(Detection).count()
        
        # Get most recently seen person
        recent_person = db.query(Person).order_by(Person.last_seen.desc()).first()
        
        # Get most detected person
        most_detected = db.query(Person).order_by(Person.total_detections.desc()).first()
        
        return {
            "total_persons": total_persons,
            "total_detections": total_detections,
            "most_recent_person": {
                "person_id": recent_person.person_id,
                "last_seen": recent_person.last_seen
            } if recent_person else None,
            "most_detected_person": {
                "person_id": most_detected.person_id,
                "total_detections": most_detected.total_detections
            } if most_detected else None
        }
    except Exception as e:
        print(f"Error in get_system_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")