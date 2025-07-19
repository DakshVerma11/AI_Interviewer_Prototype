# config.py
import os

# Base directory for the application data
BASE_DIR = 'AI-Interviewer'

# Path to the CSV file storing login details
LOGIN_DETAILS_CSV = os.path.join(BASE_DIR, 'logindetails.csv')

# Path to the folder storing user-specific data
USERS_FOLDER = os.path.join(BASE_DIR, 'Users')

# Hugging Face token for audio diarization and transcription
# IMPORTANT: Replace with your actual Hugging Face token or load from environment variables
HF_TOKEN = "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" 

# Create necessary directories if they don't exist
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(USERS_FOLDER, exist_ok=True)

# Initialize login details CSV if it doesn't exist
if not os.path.exists(LOGIN_DETAILS_CSV):
    with open(LOGIN_DETAILS_CSV, 'w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['username', 'email', 'password', 'created_date'])