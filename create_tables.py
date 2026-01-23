from database import engine, Base
from models import *
from sqlalchemy import inspect, text

def check_column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table"""
    try:
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except Exception as e:
        print(f"✗ Error checking column: {e}")
        return False

def add_role_column_if_missing():
    """Add role column to existing persons table if it doesn't exist"""
    try:
        # Check if persons table exists
        inspector = inspect(engine)
        if 'persons' not in inspector.get_table_names():
            print("  ℹ️  Persons table doesn't exist yet, will be created with role column")
            return True
        
        # Check if role column exists
        if check_column_exists('persons', 'role'):
            print("  ✓ Role column already exists in persons table")
            return True
        
        print("  ⚠️  Role column missing, adding it now...")
        
        # Add the role column with default value 'Customer'
        with engine.connect() as conn:
            # For SQLite
            if 'sqlite' in str(engine.url):
                conn.execute(text(
                    "ALTER TABLE persons ADD COLUMN role VARCHAR(20) DEFAULT 'Customer' NOT NULL"
                ))
            # For MySQL
            else:
                conn.execute(text(
                    "ALTER TABLE persons ADD COLUMN role VARCHAR(20) DEFAULT 'Customer' NOT NULL"
                ))
            conn.commit()
        
        print("  ✓ Role column added successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Error adding role column: {str(e)}")
        return False

def create_tables():
    print("\n" + "="*60)
    print("DATABASE INITIALIZATION")
    print("="*60)
    
    try:
        # Check if we need to add role column to existing table
        add_role_column_if_missing()
        
        # Create all tables (will skip existing ones)
        print("\nCreating database tables...")
        Base.metadata.create_all(bind=engine)
        
        print("✓ Tables created successfully!")
        print("\nCreated tables:")
        print("  - persons (with role: Employee/Customer)")
        print("  - detections")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"✗ Error creating tables: {str(e)}")
        print("="*60 + "\n")
        return False
    
    return True

def drop_tables():
    print("\n" + "="*60)
    print("DROPPING ALL TABLES")
    print("="*60)
    
    try:
        Base.metadata.drop_all(bind=engine)
        print("✓ Tables dropped successfully!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"✗ Error dropping tables: {str(e)}")
        print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--drop":
        drop_tables()
    
    create_tables()