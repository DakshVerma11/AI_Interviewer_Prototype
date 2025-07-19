# utils/user_manager.py
import os
import csv
import hashlib
import json
from datetime import datetime
from config import LOGIN_DETAILS_CSV, USERS_FOLDER

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hashlib.sha256(password.encode()).hexdigest() == hashed

def create_user_folder(username):
    """Create user folder structure"""
    user_dir = os.path.join(USERS_FOLDER, username)
    interview_dir = os.path.join(user_dir, 'interview')
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(interview_dir, exist_ok=True)
    return user_dir

def check_user_exists(username, email):
    """Check if user already exists"""
    if not os.path.exists(LOGIN_DETAILS_CSV):
        return False
    
    with open(LOGIN_DETAILS_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['username'].lower() == username.lower() or row['email'].lower() == email.lower():
                return True
    return False

def authenticate_user(username, password):
    """Authenticate user login"""
    if not os.path.exists(LOGIN_DETAILS_CSV):
        return False
    
    with open(LOGIN_DETAILS_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['username'].lower() == username.lower():
                return verify_password(password, row['password'])
    return False

def get_user_info(username):
    """Get user information"""
    if not os.path.exists(LOGIN_DETAILS_CSV):
        return None
    
    with open(LOGIN_DETAILS_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['username'].lower() == username.lower():
                return {
                    'username': row['username'],
                    'email': row['email'],
                    'created_date': row['created_date']
                }
    return None

