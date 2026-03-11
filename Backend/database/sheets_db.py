import gspread
from google.oauth2.service_account import Credentials
import hashlib
import json
import uuid
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class GoogleSheetsDB:
    def __init__(self):
        scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']
        
        # Use environment variables for credentials
        credentials_info = {
            "type": "service_account",
            "project_id": os.getenv('GOOGLE_PROJECT_ID'),
            "private_key_id": os.getenv('GOOGLE_PRIVATE_KEY_ID'),
            "private_key": os.getenv('GOOGLE_PRIVATE_KEY').replace('\\n', '\n'),
            "client_email": os.getenv('GOOGLE_CLIENT_EMAIL'),
            "client_id": os.getenv('GOOGLE_CLIENT_ID'),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv('GOOGLE_CLIENT_X509_CERT_URL')
        }
        
        creds = Credentials.from_service_account_info(credentials_info, scopes=scope)
        self.client = gspread.authorize(creds)
        self.workbook = self.client.open_by_key(os.getenv('GOOGLE_SHEET_ID'))
        
        # Initialize all sheets
        self.user_sheet = self._get_or_create_sheet('User', ['ID', 'Name', 'Username', 'Password', 'Role'])
        self.map_sheet = self._get_or_create_sheet('Map', ['UID', 'Longitude', 'Latitude', 'CatID'])
        self.meowdex_sheet = self._get_or_create_sheet('MeowDex', ['CatID', 'CatName', 'CatPersonal', 'Cat'])
        self.history_sheet = self._get_or_create_sheet('History', ['ID', 'CatID'])
        self.personal_sheet = self._get_or_create_sheet('Personal', ['Match_ID', 'Housing', 'Lifestyle', 'Personality', 'User_Style', 'Recommended_Cat', 'Why_Match'])
        self.user_personal_sheet = self._get_or_create_sheet('UserPersonal', ['UID', 'Housing', 'Lifestyle', 'Personality', 'Recommended_Cat', 'Match_ID'])
    
    def _get_or_create_sheet(self, sheet_name, headers):
        try:
            sheet = self.workbook.worksheet(sheet_name)
        except:
            sheet = self.workbook.add_worksheet(title=sheet_name, rows=1000, cols=len(headers))
            sheet.append_row(headers)
        return sheet
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, name, username, password, role='user'):
        hashed_password = self.hash_password(password)
        user_id = str(uuid.uuid4())
        
        # Check if username exists
        users = self.user_sheet.get_all_records()
        for user in users:
            if user.get('Username') == username:
                return {'error': 'Username already exists'}
        
        # Add new user with role
        self.user_sheet.append_row([user_id, name, username, hashed_password, role])
        return {'success': 'User registered successfully', 'user_id': user_id}
    
    def verify_user(self, username, password):
        hashed_password = self.hash_password(password)
        users = self.user_sheet.get_all_records()
        
        for user in users:
            if user.get('Username') == username and user.get('Password') == hashed_password:
                return {
                    'success': True, 
                    'user_id': user.get('ID'), 
                    'name': user.get('Name'),
                    'role': user.get('Role', 'user')
                }
        
        return {'success': False, 'error': 'Invalid credentials'}
    
    def add_map_location(self, uid, longitude, latitude, cat_id):
        self.map_sheet.append_row([uid, longitude, latitude, cat_id])
        return {'success': 'Location added'}
    
    def add_history(self, cat_id):
        history_id = str(uuid.uuid4())
        self.history_sheet.append_row([history_id, cat_id])
        return {'success': 'History added', 'history_id': history_id}
    
    def save_user_personal(self, uid, housing, lifestyle, personality, recommended_cat, match_id):
        self.user_personal_sheet.append_row([uid, housing, lifestyle, personality, recommended_cat, match_id])
        return {'success': 'User personal data saved'}
    
    def get_cat_recommendation(self, housing, lifestyle, personality):
        # Simple matching logic
        match_id = str(uuid.uuid4())
        
        if housing == 'A' and lifestyle == 'A' and personality == 'A':
            recommended_cat = 'Persian'
            why_match = 'Calm breed suitable for condo living'
        elif housing == 'B' and lifestyle == 'B' and personality == 'B':
            recommended_cat = 'Maine Coon'
            why_match = 'Active breed good for house with limited time'
        else:
            recommended_cat = 'British Shorthair'
            why_match = 'Adaptable breed suitable for various lifestyles'
        
        # Save to Personal sheet
        self.personal_sheet.append_row([match_id, housing, lifestyle, personality, '', recommended_cat, why_match])
        
        return {
            'match_id': match_id,
            'recommended_cat': recommended_cat,
            'why_match': why_match
        }