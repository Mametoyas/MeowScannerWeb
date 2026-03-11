from sheets_db import GoogleSheetsDB
import os
from dotenv import load_dotenv

load_dotenv()

def test_database_operations():
    try:
        print("1. Testing database connection...")
        db = GoogleSheetsDB()
        print("✓ Database connected successfully")
        
        # Test add_history
        print("\n2. Testing add_history...")
        result = db.add_history("U00001", "Class_7")
        print(f"Add history result: {result}")
        
        # Test add_map_location  
        print("\n3. Testing add_map_location...")
        result = db.add_map_location("U00001", 100.523186, 13.736717, "Class_7")
        print(f"Add location result: {result}")
        
        # Check if data was actually added
        print("\n4. Checking History sheet...")
        history_records = db.history_sheet.get_all_records()
        print(f"History records count: {len(history_records)}")
        if history_records:
            print(f"Last history record: {history_records[-1]}")
        
        print("\n5. Checking Map sheet...")
        map_records = db.map_sheet.get_all_records()
        print(f"Map records count: {len(map_records)}")
        if map_records:
            print(f"Last map record: {map_records[-1]}")
            
        print("\n✓ All tests completed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_operations()