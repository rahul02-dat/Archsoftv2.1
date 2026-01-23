from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from schema import DetectionRequest, DetectionResponse, PersonListResponse, SingleFaceDetection, BoundingBox
from models import Person, Detection
from utils import face_recognition_system
from datetime import datetime, timedelta
import pytz
import random
import string

detection_router = APIRouter(prefix="/detection", tags=["Detection"])

IST = pytz.timezone('Asia/Kolkata')

# SECURITY: Strict thresholds for role upgrade
ROLE_UPGRADE_UNIFORM_THRESHOLD = 0.7  # Require high uniform confidence
ROLE_UPGRADE_FACE_CONFIDENCE = 0.6    # Require high face match confidence

def generate_person_id(db: Session) -> str:
    """
    Generate a random person ID in format: PersonID:ABC123XYZ
    Ensures uniqueness by checking against existing IDs
    """
    while True:
        letters1 = ''.join(random.choices(string.ascii_uppercase, k=3))
        numbers = ''.join(random.choices(string.digits, k=3))
        letters2 = ''.join(random.choices(string.ascii_uppercase, k=3))
        
        random_id = f"{letters1}{numbers}{letters2}"
        person_id = f"PersonID:{random_id}"
        
        existing = db.query(Person).filter(Person.person_id == person_id).first()
        if not existing:
            return person_id

@detection_router.post("/recognize", response_model=DetectionResponse)
async def recognize_face(
    request: DetectionRequest,
    db: Session = Depends(get_db)
):
    try:
        face_results, message = face_recognition_system.process_detection_image(request.image_base64)
        
        if face_results is None:
            raise HTTPException(status_code=400, detail=message)
        
        persons = db.query(Person).all()
        stored_embeddings = [person.embedding for person in persons] if persons else []
        
        detections = []
        
        for face_data in face_results:
            embedding = face_data['embedding']
            bbox = face_data['bbox']
            is_employee = face_data['is_employee']
            uniform_score = face_data['uniform_score']
            
            # Determine role based on uniform detection
            detected_role = "Employee" if is_employee else "Customer"
            
            is_match = False
            matched_person = None
            confidence = 1.0
            should_increment = request.increment_detection
            
            # This will hold the role to return in THIS detection response
            current_session_role = detected_role
            
            if stored_embeddings:
                is_match, confidence, match_idx = face_recognition_system.verify_face(
                    embedding, 
                    stored_embeddings
                )
                
                if is_match:
                    matched_person = persons[match_idx]
                    
                    current_time = datetime.now(IST)
                    
                    # FIXED: Privilege Escalation Vulnerability
                    # We now require BOTH high uniform_score AND high face confidence
                    # before permanently upgrading a Customer to Employee in the database
                    should_upgrade_role = (
                        matched_person.role == "Customer" and 
                        detected_role == "Employee" and
                        uniform_score >= ROLE_UPGRADE_UNIFORM_THRESHOLD and
                        confidence >= ROLE_UPGRADE_FACE_CONFIDENCE
                    )
                    
                    if should_upgrade_role:
                        matched_person.role = "Employee"
                        print(f"✓ Role upgraded: {matched_person.person_id} Customer -> Employee")
                        print(f"  - Uniform score: {uniform_score:.3f} (threshold: {ROLE_UPGRADE_UNIFORM_THRESHOLD})")
                        print(f"  - Face confidence: {confidence:.3f} (threshold: {ROLE_UPGRADE_FACE_CONFIDENCE})")
                    
                    # For the current session, use the detected role regardless of database state
                    # This allows temporary employee detection without permanent database changes
                    current_session_role = detected_role
                    
                    # Update last seen time
                    last_seen = matched_person.last_seen
                    if last_seen.tzinfo is None:
                        last_seen = IST.localize(last_seen)
                    
                    time_since_last_seen = current_time - last_seen
                    should_increment_for_this_person = time_since_last_seen > timedelta(hours=24)
                    
                    matched_person.last_seen = current_time
                    
                    if should_increment and should_increment_for_this_person:
                        matched_person.total_detections += 1
                        
                        old_avg = matched_person.average_confidence
                        old_count = matched_person.total_detections - 1
                        new_avg = ((old_avg * old_count) + confidence) / matched_person.total_detections
                        matched_person.average_confidence = new_avg
                    
                    if should_increment and should_increment_for_this_person:
                        detection_record = Detection(
                            person_id=matched_person.person_id,
                            detection_time=current_time,
                            confidence_score=confidence,
                            location=request.location or "Unknown"
                        )
                        
                        db.add(detection_record)
            
            if not is_match:
                new_person_id = generate_person_id(db)
                embedding_bytes = face_recognition_system.serialize_embedding(embedding)
                
                new_person = Person(
                    person_id=new_person_id,
                    first_seen=datetime.now(IST),
                    last_seen=datetime.now(IST),
                    total_detections=1,
                    average_confidence=1.0,
                    embedding=embedding_bytes,
                    role=detected_role  # Set role based on uniform detection
                )
                
                db.add(new_person)
                
                detection_record = Detection(
                    person_id=new_person_id,
                    detection_time=datetime.now(IST),
                    confidence_score=1.0,
                    location=request.location or "Unknown"
                )
                
                db.add(detection_record)
                
                matched_person = new_person
                confidence = 1.0
                current_session_role = detected_role
            
            db.flush()
            db.refresh(matched_person)
            
            # Return the current session role (which may differ from database role)
            detections.append(SingleFaceDetection(
                is_match=is_match,
                person_id=matched_person.person_id,
                confidence_score=confidence,
                first_seen=matched_person.first_seen,
                last_seen=matched_person.last_seen,
                total_detections=matched_person.total_detections,
                bbox=BoundingBox(**bbox),
                role=current_session_role  # Use session role, not necessarily DB role
            ))
        
        db.commit()
        
        return DetectionResponse(
            success=True,
            faces_detected=len(detections),
            detections=detections,
            message=f"Detected {len(detections)} face(s)"
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
                    average_confidence=person.average_confidence,
                    role=person.role
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
        "role": person.role,
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
    try:
        person = db.query(Person).filter(Person.person_id == person_id).first()
        
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        db.query(Detection).filter(Detection.person_id == person_id).delete()
        
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
    try:
        db.query(Detection).delete()
        db.query(Person).delete()
        db.commit()
        
        return {"message": "All data reset successfully"}
    except Exception as e:
        db.rollback()
        print(f"Error in reset_all_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting data: {str(e)}")

@detection_router.get("/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    try:
        total_persons = db.query(Person).count()
        total_detections = db.query(Detection).count()
        total_employees = db.query(Person).filter(Person.role == "Employee").count()
        total_customers = db.query(Person).filter(Person.role == "Customer").count()
        
        recent_person = db.query(Person).order_by(Person.last_seen.desc()).first()
        most_detected = db.query(Person).order_by(Person.total_detections.desc()).first()
        
        return {
            "total_persons": total_persons,
            "total_detections": total_detections,
            "total_employees": total_employees,
            "total_customers": total_customers,
            "most_recent_person": {
                "person_id": recent_person.person_id,
                "last_seen": recent_person.last_seen,
                "role": recent_person.role
            } if recent_person else None,
            "most_detected_person": {
                "person_id": most_detected.person_id,
                "total_detections": most_detected.total_detections,
                "role": most_detected.role
            } if most_detected else None
        }
    except Exception as e:
        print(f"Error in get_system_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")