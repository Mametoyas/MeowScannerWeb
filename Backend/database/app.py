import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from sheets_db import GoogleSheetsDB

app = Flask(__name__)
CORS(app)

# Initialize Google Sheets DB (lazy loading)
sheets_db = None

def get_sheets_db():
    global sheets_db
    if sheets_db is None:
        sheets_db = GoogleSheetsDB()
    return sheets_db

@app.route("/", methods=["GET"])
def mainRoute():
    return "Database API Service"

@app.route("/health", methods=["GET"])
def check_health():
    try:
        # Test database connection
        db = get_sheets_db()
        return jsonify({"code": 200, "status": "healthy", "database": "connected"})
    except Exception as e:
        return jsonify({"code": 500, "status": "unhealthy", "error": str(e)}), 500

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get('name')
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')  # Default role is 'user'
    
    if not all([name, username, password]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    db = get_sheets_db()
    result = db.register_user(name, username, password, role)
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not all([username, password]):
        return jsonify({'error': 'Missing username or password'}), 400
    
    db = get_sheets_db()
    result = db.verify_user(username, password)
    
    if not result['success']:
        return jsonify(result), 401
    
    return jsonify(result), 200

@app.route("/cat-recommendation", methods=["POST"])
def cat_recommendation():
    data = request.get_json()
    housing = data.get('location')  # A, B, C
    lifestyle = data.get('time')    # A, B, C  
    personality = data.get('personality')  # A, B, C
    uid = data.get('uid')
    
    if not all([housing, lifestyle, personality]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    db = get_sheets_db()
    # Get recommendation from database
    result = db.get_cat_recommendation(housing, lifestyle, personality)
    
    # Save user personal data if uid provided
    if uid and result.get('match_id') != 'M000':
        try:
            # Save to UserPersonal sheet
            user_personal_id = db._generate_sequential_id(db.user_personal_sheet, 'UP')
            db.user_personal_sheet.append_row([
                user_personal_id,
                uid, 
                housing, 
                lifestyle, 
                personality, 
                result['recommended_cat'], 
                result['match_id']
            ])
        except Exception as e:
            print(f"Failed to save user personal data: {e}")
    
    return jsonify(result), 200

@app.route("/add-location", methods=["POST"])
def add_location():
    data = request.get_json()
    uid = data.get('uid')
    longitude = data.get('longitude')
    latitude = data.get('latitude')
    cat_id = data.get('cat_id')
    
    if not all([uid, longitude, latitude, cat_id]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    db = get_sheets_db()
    result = db.add_map_location(uid, longitude, latitude, cat_id)
    return jsonify(result), 200

@app.route("/add-history", methods=["POST"])
def add_history():
    data = request.get_json()
    user_id = data.get('user_id')
    cat_id = data.get('cat_id')
    
    if not all([user_id, cat_id]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    db = get_sheets_db()
    result = db.add_history(user_id, cat_id)
    return jsonify(result), 200

@app.route("/get-cats", methods=["GET"])
def get_cats():
    db = get_sheets_db()
    result = db.get_all_cats()
    return jsonify(result), 200

@app.route("/add-cat", methods=["POST"])
def add_cat():
    data = request.get_json()
    cat_id = data.get('CatID')
    cat_name = data.get('CatName')
    cat_personal = data.get('CatPersonal')
    cat_details = data.get('CatDetails')  # Changed from 'Cat' to 'CatDetails'
    prices = data.get('Prices')
    img_url = data.get('ImgURL')
    
    if not all([cat_id, cat_name, cat_personal, cat_details]):
        return jsonify({'error': 'CatID, CatName, CatPersonal, and CatDetails are required'}), 400
    
    db = get_sheets_db()
    result = db.add_cat(cat_id, cat_name, cat_personal, cat_details, prices, img_url)
    return jsonify(result), 200

@app.route("/update-cat", methods=["POST"])
def update_cat():
    data = request.get_json()
    cat_id = data.get('CatID')
    cat_name = data.get('CatName')
    cat_personal = data.get('CatPersonal')
    cat_details = data.get('CatDetails')  # Changed from 'Cat' to 'CatDetails'
    prices = data.get('Prices')
    img_url = data.get('ImgURL')
    
    if not all([cat_id, cat_name, cat_personal, cat_details]):
        return jsonify({'error': 'CatID, CatName, CatPersonal, and CatDetails are required'}), 400
    
    db = get_sheets_db()
    result = db.update_cat(cat_id, cat_name, cat_personal, cat_details, prices, img_url)
    return jsonify(result), 200

@app.route("/delete-cat", methods=["POST"])
def delete_cat():
    data = request.get_json()
    cat_id = data.get('cat_id')
    
    if not cat_id:
        return jsonify({'error': 'Cat ID is required'}), 400
    
    db = get_sheets_db()
    result = db.delete_cat(cat_id)
    return jsonify(result), 200

@app.route("/get-users", methods=["GET"])
def get_users():
    db = get_sheets_db()
    result = db.get_all_users()
    return jsonify(result), 200

@app.route("/add-user", methods=["POST"])
def add_user():
    data = request.get_json()
    user_id = data.get('ID')
    name = data.get('Name')
    username = data.get('Username')
    password = data.get('Password')
    role = data.get('Role', 'user')
    
    if not all([user_id, name, username, password]):
        return jsonify({'error': 'All fields are required'}), 400
    
    db = get_sheets_db()
    result = db.add_user(user_id, name, username, password, role)
    return jsonify(result), 200

@app.route("/update-user", methods=["POST"])
def update_user():
    data = request.get_json()
    user_id = data.get('ID')
    name = data.get('Name')
    username = data.get('Username')
    password = data.get('Password')
    role = data.get('Role')
    
    if not all([user_id, name, username, role]):
        return jsonify({'error': 'ID, Name, Username, and Role are required'}), 400
    
    db = get_sheets_db()
    result = db.update_user(user_id, name, username, password, role)
    return jsonify(result), 200

@app.route("/add-test-data", methods=["POST"])
def add_test_data():
    """Add test data for debugging"""
    try:
        db = get_sheets_db()
        # Add test location data
        test_locations = [
            {"uid": "U00001", "longitude": 100.523186, "latitude": 13.736717, "cat_id": "C0001"},
            {"uid": "U00001", "longitude": 100.533186, "latitude": 13.746717, "cat_id": "C0002"},
            {"uid": "U00001", "longitude": 100.513186, "latitude": 13.726717, "cat_id": "C0003"}
        ]
        
        results = []
        for location in test_locations:
            result = db.add_map_location(
                location["uid"], 
                location["longitude"], 
                location["latitude"], 
                location["cat_id"]
            )
            results.append(result)
        
        return jsonify({"success": True, "results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-user-map-locations", methods=["GET"])
def get_user_map_locations():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    db = get_sheets_db()
    result = db.get_user_map_locations(user_id)
    return jsonify(result), 200

@app.route("/get-map-locations", methods=["GET"])
def get_map_locations():
    db = get_sheets_db()
    result = db.get_all_map_locations()
    return jsonify(result), 200

@app.route("/delete-user", methods=["POST"])
def delete_user():
    data = request.get_json()
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    db = get_sheets_db()
    result = db.delete_user(user_id)
    return jsonify(result), 200

@app.route("/get-user-stats", methods=["GET"])
def get_user_stats():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    db = get_sheets_db()
    result = db.get_user_stats(user_id)
    return jsonify(result), 200

@app.route("/get-cat-by-name", methods=["GET"])
def get_cat_by_name():
    cat_name = request.args.get('cat_name')
    if not cat_name:
        return jsonify({'error': 'Cat name is required'}), 400
    
    db = get_sheets_db()
    # Get all cats and find by name
    cats_result = db.get_all_cats()
    
    if cats_result.get('success') and cats_result.get('cats'):
        # Search for cat by name (case insensitive)
        cat = next((c for c in cats_result['cats'] 
                   if c.get('CatName', '').lower() == cat_name.lower()), None)
        
        if cat:
            return jsonify({'success': True, 'cat': cat}), 200
    
    return jsonify({'success': False, 'message': 'Cat not found'}), 404

@app.route("/get-cat-by-id", methods=["GET"])
def get_cat_by_id():
    cat_id = request.args.get('cat_id')
    if not cat_id:
        return jsonify({'error': 'Cat ID is required'}), 400
    
    db = get_sheets_db()
    # Get all cats and find by ID
    cats_result = db.get_all_cats()
    
    if cats_result.get('success') and cats_result.get('cats'):
        # Search for cat by ID
        cat = next((c for c in cats_result['cats'] 
                   if c.get('CatID') == cat_id), None)
        
        if cat:
            return jsonify({'success': True, 'cat': cat}), 200
    
    return jsonify({'success': False, 'message': 'Cat not found'}), 404

if __name__ == '__main__':
    app.run(debug=False)