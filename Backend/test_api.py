import requests
import json

# Test database API endpoints
BASE_URL = "http://localhost:5001"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Response: {response.status_code}")
        print(f"Response Data: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health: {e}")
        return False

def test_get_map_locations():
    """Test getting map locations"""
    try:
        response = requests.get(f"{BASE_URL}/get-map-locations")
        print(f"Get Map Locations Response: {response.status_code}")
        print(f"Response Data: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing get-map-locations: {e}")
        return False

def test_add_location():
    """Test adding a location to the database"""
    data = {
        "uid": "U00001",
        "longitude": 100.523186,
        "latitude": 13.736717,
        "cat_id": "C0001"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/add-location", json=data)
        print(f"Add Location Response: {response.status_code}")
        print(f"Response Data: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing add-location: {e}")
        return False

def test_add_history():
    """Test adding history to the database"""
    data = {
        "user_id": "U00001",
        "cat_id": "C0001"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/add-history", json=data)
        print(f"Add History Response: {response.status_code}")
        print(f"Response Data: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing add-history: {e}")
        return False

def test_add_test_data():
    """Test adding test data"""
    try:
        response = requests.post(f"{BASE_URL}/add-test-data")
        print(f"Add Test Data Response: {response.status_code}")
        print(f"Response Data: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing add-test-data: {e}")
        return False

if __name__ == "__main__":
    print("Testing Database API Endpoints...")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("Database API is not running! Start it with: cd Backend/database && python app.py")
        exit(1)
    
    print()
    print("Getting current map locations:")
    test_get_map_locations()
    
    print()
    print("Adding test location:")
    test_add_location()
    
    print()
    print("Adding test history:")
    test_add_history()
    
    print()
    print("Adding multiple test data:")
    test_add_test_data()
    
    print()
    print("Getting updated map locations:")
    test_get_map_locations()
    
    print()
    print("Test completed!")