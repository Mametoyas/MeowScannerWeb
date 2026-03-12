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
        self.map_sheet = self._get_or_create_sheet('Map', ['ID', 'UID', 'Longitude', 'Latitude', 'CatID'])
        self.meowdex_sheet = self._get_or_create_sheet('MeowDex', ['CatID', 'CatName', 'CatPersonal', 'CatDetails', 'Prices(THB)', 'ImgURL'])
        self.history_sheet = self._get_or_create_sheet('History', ['ID', 'UserID', 'CatID', 'Timestamp'])
        self.personal_sheet = self._get_or_create_sheet('Personal', ['Match_ID', 'Housing', 'Lifestyle', 'Personality', 'User_Style', 'Recommended_Cat', 'Why_Match'])
        self.user_personal_sheet = self._get_or_create_sheet('UserPersonal', ['ID', 'UID', 'Housing', 'Lifestyle', 'Personality', 'Recommended_Cat', 'Match_ID'])
    
    def _generate_sequential_id(self, sheet, prefix=''):
        """Generate sequential ID starting from 00001"""
        try:
            records = sheet.get_all_records()
            if not records:
                return f"{prefix}00001"
            
            # Find the highest ID number
            max_id = 0
            for record in records:
                id_value = record.get('ID', '')
                if isinstance(id_value, str) and id_value.startswith(prefix):
                    try:
                        num = int(id_value.replace(prefix, ''))
                        max_id = max(max_id, num)
                    except ValueError:
                        continue
            
            next_id = max_id + 1
            return f"{prefix}{next_id:05d}"
        except Exception:
            return f"{prefix}00001"
    
    def _get_or_create_sheet(self, sheet_name, headers):
        try:
            sheet = self.workbook.worksheet(sheet_name)
        except:
            sheet = self.workbook.add_worksheet(title=sheet_name, rows=1000, cols=len(headers))
            sheet.append_row(headers)
        return sheet
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
        user_id = self._generate_sequential_id(self.user_sheet, 'U')
        
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
        map_id = self._generate_sequential_id(self.map_sheet, 'M')
        self.map_sheet.append_row([map_id, uid, longitude, latitude, cat_id])
        return {'success': 'Location added', 'map_id': map_id}
    
    def add_history(self, user_id, cat_id):
        history_id = self._generate_sequential_id(self.history_sheet, 'H')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.history_sheet.append_row([history_id, user_id, cat_id, timestamp])
        return {'success': 'History added', 'history_id': history_id}
    
    def save_user_personal(self, uid, housing, lifestyle, personality, recommended_cat, match_id):
        self.user_personal_sheet.append_row([uid, housing, lifestyle, personality, recommended_cat, match_id])
        return {'success': 'User personal data saved'}
    
    def get_cat_recommendation(self, housing, lifestyle, personality):
        # Get all personal data records
        personal_records = self.personal_sheet.get_all_records()
        
        # Find matching record based on user answers
        for record in personal_records:
            record_housing = record.get('Housing', '').strip()
            record_lifestyle = record.get('Lifestyle', '').strip()
            record_personality = record.get('Personality', '').strip()
            
            # Match the pattern (A, B, C)
            if (housing in record_housing and 
                lifestyle in record_lifestyle and 
                personality in record_personality):
                
                return {
                    'match_id': record.get('Match_ID'),
                    'recommended_cat': record.get('Recommended_Cat'),
                    'why_match': record.get('Why_Match'),
                    'user_style': record.get('User_Style', '')
                }
        
        # Default fallback if no match found
        return {
            'match_id': 'M000',
            'recommended_cat': 'Mixed Breed',
            'why_match': 'A versatile and loving companion suitable for various lifestyles'
        }
    
    def get_all_cats(self):
        """Get all cats from MeowDex sheet"""
        try:
            records = self.meowdex_sheet.get_all_records()
            return {'success': True, 'cats': records}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def add_cat(self, cat_id, cat_name, cat_personal, cat_details):
        """Add new cat to MeowDex sheet"""
        try:
            # Check if cat ID already exists
            records = self.meowdex_sheet.get_all_records()
            for record in records:
                if record.get('CatID') == cat_id:
                    return {'success': False, 'error': 'Cat ID already exists'}
            
            # Add new cat
            self.meowdex_sheet.append_row([cat_id, cat_name, cat_personal, cat_details])
            return {'success': True, 'message': 'Cat added successfully'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def update_cat(self, cat_id, cat_name, cat_personal, cat_details):
        """Update existing cat in MeowDex sheet"""
        try:
            records = self.meowdex_sheet.get_all_records()
            for i, record in enumerate(records, start=2):  # Start from row 2 (skip header)
                if record.get('CatID') == cat_id:
                    # Update the row
                    self.meowdex_sheet.update(f'A{i}:D{i}', [[cat_id, cat_name, cat_personal, cat_details]])
                    return {'success': True, 'message': 'Cat updated successfully'}
            
            return {'success': False, 'error': 'Cat not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def delete_cat(self, cat_id):
        """Delete cat from MeowDex sheet"""
        try:
            records = self.meowdex_sheet.get_all_records()
            for i, record in enumerate(records, start=2):  # Start from row 2 (skip header)
                if record.get('CatID') == cat_id:
                    # Delete the row
                    self.meowdex_sheet.delete_rows(i)
                    return {'success': True, 'message': 'Cat deleted successfully'}
            
            return {'success': False, 'error': 'Cat not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_all_users(self):
        """Get all users from User sheet"""
        try:
            records = self.user_sheet.get_all_records()
            return {'success': True, 'users': records}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def add_user(self, user_id, name, username, password, role='user'):
        """Add new user to User sheet"""
        try:
            # Check if user ID or username already exists
            records = self.user_sheet.get_all_records()
            for record in records:
                if record.get('ID') == user_id:
                    return {'success': False, 'error': 'User ID already exists'}
                if record.get('Username') == username:
                    return {'success': False, 'error': 'Username already exists'}
            
            # Hash password and add new user
            hashed_password = self.hash_password(password)
            self.user_sheet.append_row([user_id, name, username, hashed_password, role])
            return {'success': True, 'message': 'User added successfully'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def update_user(self, user_id, name, username, password, role):
        """Update existing user in User sheet"""
        try:
            records = self.user_sheet.get_all_records()
            for i, record in enumerate(records, start=2):  # Start from row 2 (skip header)
                if record.get('ID') == user_id:
                    # Check if username is taken by another user
                    for j, other_record in enumerate(records, start=2):
                        if j != i and other_record.get('Username') == username:
                            return {'success': False, 'error': 'Username already exists'}
                    
                    # Keep existing password if new password is empty
                    if password:
                        hashed_password = self.hash_password(password)
                    else:
                        hashed_password = record.get('Password')
                    
                    # Update the row
                    self.user_sheet.update(f'A{i}:E{i}', [[user_id, name, username, hashed_password, role]])
                    return {'success': True, 'message': 'User updated successfully'}
            
            return {'success': False, 'error': 'User not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def delete_user(self, user_id):
        """Delete user from User sheet"""
        try:
            records = self.user_sheet.get_all_records()
            for i, record in enumerate(records, start=2):  # Start from row 2 (skip header)
                if record.get('ID') == user_id:
                    # Delete the row
                    self.user_sheet.delete_rows(i)
                    return {'success': True, 'message': 'User deleted successfully'}
            
            return {'success': False, 'error': 'User not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_all_map_locations(self):
        """Get all map locations from Map sheet"""
        try:
            records = self.map_sheet.get_all_records()
            return {'success': True, 'locations': records}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user_map_locations(self, user_id):
        """Get map locations for specific user from Map sheet"""
        try:
            records = self.map_sheet.get_all_records()
            user_locations = [record for record in records if record.get('UID') == user_id]
            return {'success': True, 'locations': user_locations}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user_stats(self, user_id):
        """Get user statistics from History and Map sheets"""
        try:
            # Get history records for user
            history_records = self.history_sheet.get_all_records()
            user_history = [record for record in history_records if record.get('UserID') == user_id]
            
            # Get map locations for user
            map_records = self.map_sheet.get_all_records()
            user_locations = [record for record in map_records if record.get('UID') == user_id]
            
            # Count unique cats discovered
            unique_cats = set()
            for record in user_history:
                cat_id = record.get('CatID')
                if cat_id:
                    unique_cats.add(cat_id)
            
            stats = {
                'predictions_made': len(user_history),
                'cats_discovered': len(unique_cats),
                'locations_mapped': len(user_locations)
            }
            
            return {'success': True, 'stats': stats}
        except Exception as e:
            return {'success': False, 'error': str(e)}