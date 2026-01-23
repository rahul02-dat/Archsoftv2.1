"""
Database Migration Script - Add 'role' column to existing Person table
MySQL Compatible Version

Usage:
    python migrate_database_mysql.py
"""

from sqlalchemy import text, inspect
from database import engine, SessionLocal
from models import Person
import sys

def check_column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table using SQLAlchemy inspector"""
    try:
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except Exception as e:
        print(f"Error checking column: {e}")
        return False

def migrate_database():
    """Add 'role' column to Person table if it doesn't exist"""
    
    print("=" * 60)
    print("DATABASE MIGRATION: Adding 'role' column to Person table")
    print("=" * 60)
    
    try:
        # Check if role column already exists
        if check_column_exists('persons', 'role'):
            print("✓ 'role' column already exists in persons table")
            print("  No migration needed!")
            return True
        
        print("\n📋 Migration Steps:")
        print("  1. Adding 'role' column to persons table...")
        
        # Add the role column with default value 'Customer'
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE persons ADD COLUMN role VARCHAR(20) DEFAULT 'Customer' NOT NULL"
            ))
            conn.commit()
        
        print("  ✓ Column 'role' added successfully")
        
        # Verify the column was added
        if check_column_exists('persons', 'role'):
            print("\n✓ Migration completed successfully!")
            print("\nColumn Details:")
            print("  - Column Name: role")
            print("  - Type: VARCHAR(20)")
            print("  - Default: 'Customer'")
            print("  - Nullable: NO")
            
            # Show current person count
            db = SessionLocal()
            try:
                person_count = db.query(Person).count()
                print(f"\n📊 Current Database State:")
                print(f"  - Total Persons: {person_count}")
                print(f"  - All existing persons set as: Customer")
                print(f"  - New persons will be auto-detected as: Employee or Customer")
            finally:
                db.close()
            
            return True
        else:
            print("\n✗ Migration verification failed!")
            return False
            
    except Exception as e:
        print(f"\n✗ Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def rollback_migration():
    """Remove 'role' column (rollback migration)"""
    
    print("\n" + "=" * 60)
    print("ROLLBACK: Removing 'role' column from Person table")
    print("=" * 60)
    
    try:
        if not check_column_exists('persons', 'role'):
            print("✓ 'role' column doesn't exist. Nothing to rollback.")
            return True
        
        print("\n⚠️  WARNING: This will remove the 'role' column from the persons table")
        
        confirm = input("\nAre you sure you want to rollback? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Rollback cancelled.")
            return False
        
        print("\nPerforming rollback...")
        
        with engine.connect() as conn:
            # MySQL supports DROP COLUMN directly
            conn.execute(text("ALTER TABLE persons DROP COLUMN role"))
            conn.commit()
        
        print("✓ Rollback completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Rollback failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def show_current_state():
    """Show current database state"""
    print("\n" + "=" * 60)
    print("CURRENT DATABASE STATE")
    print("=" * 60)
    
    has_role = check_column_exists('persons', 'role')
    print(f"'role' column exists: {'YES ✓' if has_role else 'NO ✗'}")
    
    db = SessionLocal()
    try:
        person_count = db.query(Person).count()
        print(f"Total Persons: {person_count}")
        
        if has_role and person_count > 0:
            employee_count = db.query(Person).filter(Person.role == 'Employee').count()
            customer_count = db.query(Person).filter(Person.role == 'Customer').count()
            print(f"  - Employees: {employee_count}")
            print(f"  - Customers: {customer_count}")
    except Exception as e:
        print(f"Error reading database: {e}")
    finally:
        db.close()
    
    print("=" * 60)

if __name__ == "__main__":
    print("\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--rollback":
            success = rollback_migration()
        elif sys.argv[1] == "--status":
            show_current_state()
            sys.exit(0)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("\nUsage:")
            print("  python migrate_database_mysql.py           # Run migration")
            print("  python migrate_database_mysql.py --status  # Show current state")
            print("  python migrate_database_mysql.py --rollback # Rollback migration")
            sys.exit(1)
    else:
        success = migrate_database()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Operation completed successfully!")
    else:
        print("❌ Operation failed!")
    print("=" * 60 + "\n")
    
    sys.exit(0 if success else 1)