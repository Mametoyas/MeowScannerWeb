from sheets_db import GoogleSheetsDB
import os
from dotenv import load_dotenv

load_dotenv()

def test_database():
    try:
        print("Testing database connection...")
        db = GoogleSheetsDB()
        print("✓ Database connected successfully")
        
        # Test register user
        print("\nTesting user registration...")
        result = db.register_user("Test User", "testuser", "password123", "user")
        print(f"Register result: {result}")
        
        # Test login
        print("\nTesting user login...")
        login_result = db.verify_user("testuser", "password123")
        print(f"Login result: {login_result}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Check your .env file and Google Sheets credentials")

if __name__ == "__main__":
    test_database()