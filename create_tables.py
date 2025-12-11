from database import engine, Base
from models import *

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