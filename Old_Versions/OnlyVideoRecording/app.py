from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import os
import json
import csv
from datetime import datetime
import uuid
import hashlib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'ai_interviewer_secret_key_2025'  # Change this in production

# Configuration
BASE_DIR = 'AI-Interviewer'
LOGIN_DETAILS_CSV = os.path.join(BASE_DIR, 'logindetails.csv')
USERS_FOLDER = os.path.join(BASE_DIR, 'Users')

# Create necessary directories
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(USERS_FOLDER, exist_ok=True)

# Initialize login details CSV if it doesn't exist
if not os.path.exists(LOGIN_DETAILS_CSV):
    with open(LOGIN_DETAILS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['username', 'email', 'password', 'created_date'])

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

class InterviewSession:
    def __init__(self, session_id, username, start_time=None):
        self.session_id = session_id
        self.username = username
        self.start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00')) if start_time else datetime.now()
        self.end_time = None
        self.total_questions = 0
        self.questions_answered = 0
        self.question_timings = []
        self.recording_filename = None
        self.completed = False
        self.duration_seconds = 0
        self.role_applied = ""
        
    def to_dict(self):
        return {
            'session_id': self.session_id,
            'username': self.username,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_questions': self.total_questions,
            'questions_answered': self.questions_answered,
            'question_timings': self.question_timings,
            'recording_filename': self.recording_filename,
            'completed': self.completed,
            'duration_seconds': self.duration_seconds,
            'duration_formatted': self.format_duration(),
            'role_applied': self.role_applied
        }
    
    def format_duration(self):
        if self.duration_seconds == 0:
            return "00:00"
        minutes = self.duration_seconds // 60
        seconds = self.duration_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

# Store active sessions
active_sessions = {}

@app.route('/')
def index():
    if 'username' in session:
        return send_file('dashboard.html')
    return send_file('login.html')

@app.route('/login')
def login_page():
    return send_file('login.html')

@app.route('/register')
def register_page():
    return send_file('register.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return send_file('dashboard.html')

@app.route('/interview')
def interview_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return send_file('interview.html')

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        # Validate input
        if not all([username, email, password]):
            return jsonify({'error': 'All fields are required'}), 400
        
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        if '@' not in email:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if user already exists
        if check_user_exists(username, email):
            return jsonify({'error': 'Username or email already exists'}), 400
        
        # Hash password and save user
        hashed_password = hash_password(password)
        
        with open(LOGIN_DETAILS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([username, email, hashed_password, datetime.now().isoformat()])
        
        # Create user folder
        create_user_folder(username)
        
        return jsonify({'success': True, 'message': 'Registration successful'})
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
        if authenticate_user(username, password):
            session['username'] = username
            user_info = get_user_info(username)
            return jsonify({'success': True, 'user': user_info})
        else:
            return jsonify({'error': 'Invalid username or password'}), 401
            
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.pop('username', None)
    return jsonify({'success': True})

@app.route('/api/current-user')
def current_user():
    """Get current logged in user"""
    if 'username' in session:
        user_info = get_user_info(session['username'])
        return jsonify({'user': user_info})
    return jsonify({'user': None})

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    """Upload user resume"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        username = session['username']
        role = request.form.get('role', '').strip()
        
        if not role:
            return jsonify({'error': 'Role is required'}), 400
        
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file uploaded'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save resume
        user_dir = os.path.join(USERS_FOLDER, username)
        filename = secure_filename(f"resume_{username}.pdf")
        filepath = os.path.join(user_dir, filename)
        file.save(filepath)
        
        # Save role information
        role_info = {
            'role': role,
            'resume_filename': filename,
            'upload_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(user_dir, 'profile.json'), 'w') as f:
            json.dump(role_info, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Resume uploaded successfully'})
        
    except Exception as e:
        print(f"Resume upload error: {e}")
        return jsonify({'error': 'Failed to upload resume'}), 500

@app.route('/api/user-profile')
def user_profile():
    """Get user profile information"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        username = session['username']
        user_dir = os.path.join(USERS_FOLDER, username)
        profile_path = os.path.join(user_dir, 'profile.json')
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            return jsonify({'profile': profile})
        else:
            return jsonify({'profile': None})
            
    except Exception as e:
        print(f"Profile error: {e}")
        return jsonify({'error': 'Failed to get profile'}), 500

@app.route('/api/questions')
def get_questions():
    """Load questions from user-specific CSV file"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        username = session['username']
        user_questions_file = os.path.join(USERS_FOLDER, username, 'questions.csv')
        
        if not os.path.exists(user_questions_file):
            return jsonify({'error': 'INTERVIEW_NOT_READY'})
        
        questions = []
        with open(user_questions_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            questions = [row[0] for row in reader if row and row[0].strip()]
        
        if not questions:
            return jsonify({'error': 'INTERVIEW_NOT_READY'})
        
        return jsonify(questions)
        
    except Exception as e:
        print(f"Questions error: {e}")
        return jsonify({'error': 'Failed to load questions'}), 500

@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Initialize a new interview session"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        start_time = data.get('startTime')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        username = session['username']
        
        # Create new session
        interview_session = InterviewSession(session_id, username, start_time)
        active_sessions[session_id] = interview_session
        
        # Create session directory in user's interview folder
        session_dir = os.path.join(USERS_FOLDER, username, 'interview', session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Load total questions count from user's questions file
        user_questions_file = os.path.join(USERS_FOLDER, username, 'questions.csv')
        try:
            with open(user_questions_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                interview_session.total_questions = len([row for row in reader if row and row[0].strip()])
        except:
            return jsonify({'error': 'Questions file not found'}), 400
        
        # Get role applied for
        profile_path = os.path.join(USERS_FOLDER, username, 'profile.json')
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile = json.load(f)
                interview_session.role_applied = profile.get('role', '')
        
        # Save initial session info
        with open(os.path.join(session_dir, 'session_info.json'), 'w') as f:
            json.dump(interview_session.to_dict(), f, indent=2)
        
        print(f"Started interview session: {session_id} for user: {username}")
        return jsonify({'status': 'success', 'sessionId': session_id})
        
    except Exception as e:
        print(f"Start session error: {e}")
        return jsonify({'error': 'Failed to start session'}), 500

@app.route('/api/update-timings', methods=['POST'])
def update_timings():
    """Update question timings during the interview"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        question_timings = data.get('questionTimings', [])
        
        interview_session = active_sessions.get(session_id)
        if not interview_session:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Update session timings
        interview_session.question_timings = question_timings
        interview_session.questions_answered = len(question_timings)
        
        # Save updated session info
        username = session['username']
        session_dir = os.path.join(USERS_FOLDER, username, 'interview', session_id)
        with open(os.path.join(session_dir, 'session_info.json'), 'w') as f:
            json.dump(interview_session.to_dict(), f, indent=2)
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        print(f"Update timings error: {e}")
        return jsonify({'error': 'Failed to update timings'}), 500

@app.route('/api/save-recording', methods=['POST'])
def save_recording():
    """Save the complete interview video recording"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        session_id = request.form.get('sessionId')
        video_file = request.files.get('video')
        question_timings_str = request.form.get('questionTimings', '[]')
        end_time = request.form.get('endTime')
        
        if not all([session_id, video_file]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get session
        interview_session = active_sessions.get(session_id)
        if not interview_session:
            return jsonify({'error': 'Invalid session'}), 400
        
        username = session['username']
        
        # Parse question timings
        try:
            question_timings = json.loads(question_timings_str)
            interview_session.question_timings = question_timings
        except:
            pass
        
        # Set end time and duration
        if end_time:
            interview_session.end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            interview_session.duration_seconds = int((interview_session.end_time - interview_session.start_time).total_seconds())
        
        # Save video file
        session_dir = os.path.join(USERS_FOLDER, username, 'interview', session_id)
        filename = f"interview_{session_id}.webm"
        filepath = os.path.join(session_dir, filename)
        
        print(f"Saving recording to: {filepath}")
        video_file.save(filepath)
        
        # Update session
        interview_session.recording_filename = filename
        interview_session.questions_answered = len(interview_session.question_timings)
        
        # Save updated session info
        with open(os.path.join(session_dir, 'session_info.json'), 'w') as f:
            json.dump(interview_session.to_dict(), f, indent=2)
        
        print(f"Recording saved successfully: {filename}")
        return jsonify({'status': 'success', 'filename': filename})
        
    except Exception as e:
        print(f"Save recording error: {e}")
        return jsonify({'error': 'Failed to save recording'}), 500

@app.route('/api/finish-interview', methods=['POST'])
def finish_interview():
    """Mark interview as completed and generate final report"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        end_time = data.get('endTime')
        question_timings = data.get('questionTimings', [])
        total_duration = data.get('totalDuration', 0)
        
        interview_session = active_sessions.get(session_id)
        if not interview_session:
            return jsonify({'error': 'Invalid session'}), 400
        
        username = session['username']
        
        # Update final session data
        interview_session.completed = True
        interview_session.question_timings = question_timings
        interview_session.questions_answered = len(question_timings)
        interview_session.duration_seconds = total_duration
        
        if end_time:
            interview_session.end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Save final session info
        session_dir = os.path.join(USERS_FOLDER, username, 'interview', session_id)
        with open(os.path.join(session_dir, 'session_info.json'), 'w') as f:
            json.dump(interview_session.to_dict(), f, indent=2)
        
        # Generate comprehensive report
        generate_session_report(session_id, interview_session, username)
        
        print(f"Interview completed: {session_id} for user: {username}")
        return jsonify({'status': 'success', 'message': 'Interview completed successfully'})
        
    except Exception as e:
        print(f"Finish interview error: {e}")
        return jsonify({'error': 'Failed to finish interview'}), 500

def generate_session_report(session_id, interview_session, username):
    """Generate a comprehensive report for the interview session"""
    session_dir = os.path.join(USERS_FOLDER, username, 'interview', session_id)
    report_path = os.path.join(session_dir, 'interview_report.txt')
    
    try:
        user_info = get_user_info(username)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("AI INTERVIEWER - COMPREHENSIVE SESSION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Username: {username}\n")
            f.write(f"Email: {user_info.get('email', 'N/A')}\n")
            f.write(f"Role Applied: {interview_session.role_applied}\n")
            f.write(f"Date: {interview_session.start_time.strftime('%Y-%m-%d')}\n")
            f.write(f"Start Time: {interview_session.start_time.strftime('%H:%M:%S UTC')}\n")
            
            if interview_session.end_time:
                f.write(f"End Time: {interview_session.end_time.strftime('%H:%M:%S UTC')}\n")
            
            f.write(f"Total Duration: {interview_session.format_duration()}\n")
            f.write(f"Questions Available: {interview_session.total_questions}\n")
            f.write(f"Questions Answered: {interview_session.questions_answered}\n")
            f.write(f"Completion Rate: {(interview_session.questions_answered/interview_session.total_questions*100):.1f}%\n")
            
            if interview_session.recording_filename:
                f.write(f"Recording File: {interview_session.recording_filename}\n")
            
            f.write(f"Status: {'Completed' if interview_session.completed else 'Incomplete'}\n\n")
            
            f.write("QUESTION TIMELINE:\n")
            f.write("-" * 40 + "\n")
            
            if interview_session.question_timings:
                for timing in interview_session.question_timings:
                    time_mins = timing.get('timeFromStart', 0) // 60
                    time_secs = timing.get('timeFromStart', 0) % 60
                    f.write(f"[{time_mins:02d}:{time_secs:02d}] Q{timing.get('questionIndex', 0) + 1}: ")
                    f.write(f"{timing.get('question', 'Unknown question')}\n")
            else:
                f.write("No question timings recorded.\n")
            
            f.write("\nFILE INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write("- Recording format: WebM (VP9/Opus)\n")
            f.write("- Audio and video combined in single file\n")
            f.write("- High quality recording for analysis\n")
            f.write("- Question timings stored in JSON format\n\n")
            
            f.write("TECHNICAL DETAILS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"- System: AI Interviewer Web Platform\n")
            f.write(f"- Session Directory: {session_dir}\n")
            
        print(f"Report generated: {report_path}")
        
    except Exception as e:
        print(f"Error generating report: {e}")

@app.route('/api/user-sessions')
def list_user_sessions():
    """List all interview sessions for current user"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        username = session['username']
        sessions = []
        
        interview_dir = os.path.join(USERS_FOLDER, username, 'interview')
        if os.path.exists(interview_dir):
            for session_id in os.listdir(interview_dir):
                session_dir = os.path.join(interview_dir, session_id)
                session_info_path = os.path.join(session_dir, 'session_info.json')
                
                if os.path.exists(session_info_path):
                    with open(session_info_path, 'r') as f:
                        session_data = json.load(f)
                        
                        # Add file size information
                        recording_file = session_data.get('recording_filename')
                        if recording_file:
                            recording_path = os.path.join(session_dir, recording_file)
                            if os.path.exists(recording_path):
                                file_size_mb = os.path.getsize(recording_path) / (1024 * 1024)
                                session_data['file_size_mb'] = round(file_size_mb, 2)
                        
                        sessions.append(session_data)
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return jsonify(sessions)
        
    except Exception as e:
        print(f"List sessions error: {e}")
        return jsonify({'error': 'Failed to list sessions'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("AI INTERVIEWER WEB SERVER WITH USER MANAGEMENT")
    print("=" * 60)
    print(f"🚀 Starting server...")
    print(f"🌐 Access URL: http://localhost:5000")
    print(f"📁 Base directory: {os.path.abspath(BASE_DIR)}")
    print(f"👥 Users folder: {os.path.abspath(USERS_FOLDER)}")
    print(f"🔐 Login details: {os.path.abspath(LOGIN_DETAILS_CSV)}")
    print(f"📅 Date: 2025-06-22 12:31:38 UTC")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)