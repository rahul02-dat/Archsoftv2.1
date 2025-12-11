from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes import detection_router
from database import engine, Base
from models import *
import os

def create_tables():
    print("Creating database tables...")
    
    try:
        Base.metadata.create_all(bind=engine)
        print("✓ Tables created successfully!")
        print("\nCreated tables:")
        print("  - persons")
        print("  - detections")
        
    except Exception as e:
        print(f"✗ Error creating tables: {str(e)}")
        return False
    
    return True

def drop_tables():
    print("Dropping all tables...")
    try:
        Base.metadata.drop_all(bind=engine)
        print("✓ Tables dropped successfully!")
    except Exception as e:
        print(f"✗ Error dropping tables: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--drop":
        drop_tables()
    
    create_tables()

app = FastAPI(
    title="Automated Face Recognition System",
    description="Automated face detection and tracking system using RetinaFace detection and ArcFace embeddings",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)

# Serve static files (for production deployment)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)

@app.get("/")
async def root():
    # Serve the index.html file if it exists
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {
        "message": "Automated Face Recognition System API",
        "version": "1.0.0",
        "description": "System automatically detects and tracks faces",
        "endpoints": {
            "recognize": "/detection/recognize - Submit image for automatic recognition",
            "list_persons": "/detection/persons - List all detected persons",
            "person_history": "/detection/person/{person_id}/history - Get detection history",
            "stats": "/detection/stats - Get system statistics"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "system": "automated_face_recognition"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)