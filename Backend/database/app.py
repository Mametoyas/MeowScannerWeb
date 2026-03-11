from flask import Flask, jsonify, request
from flask_cors import CORS
from sheets_db import GoogleSheetsDB

app = Flask(__name__)
CORS(app)

# Initialize Google Sheets DB
sheets_db = GoogleSheetsDB()

@app.route("/", methods=["GET"])
def mainRoute():
    return "Database API Service"

@app.route("/health", methods=["GET"])
def check_health():
    return jsonify({"code": 200})

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get('name')
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')  # Default role is 'user'
    
    if not all([name, username, password]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    result = sheets_db.register_user(name, username, password, role)
    
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
    
    result = sheets_db.verify_user(username, password)
    
    if not result['success']:
        return jsonify(result), 401
    
    return jsonify(result), 200

@app.route("/cat-recommendation", methods=["POST"])
def cat_recommendation():
    data = request.get_json()
    housing = data.get('location')
    lifestyle = data.get('time')
    personality = data.get('personality')
    uid = data.get('uid')
    
    if not all([housing, lifestyle, personality]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    result = sheets_db.get_cat_recommendation(housing, lifestyle, personality)
    
    if uid:
        sheets_db.save_user_personal(uid, housing, lifestyle, personality, result['recommended_cat'], result['match_id'])
    
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
    
    result = sheets_db.add_map_location(uid, longitude, latitude, cat_id)
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)