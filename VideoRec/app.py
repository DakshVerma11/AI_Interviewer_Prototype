from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import os
import json
import csv
from datetime import datetime
import uuid
import hashlib
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import subprocess
import threading
import time

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

class OptimizedEyeTracker:
    def __init__(self):
        # Initialize OpenCV classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Optimized tracking parameters for 24 FPS
        self.prev_face_center = None
        self.prev_eye_centers = None
        self.face_movement_threshold = 25  # Slightly higher for 24 FPS
        self.eye_movement_threshold = 18   # Adjusted for lower frame rate
        
        # Frame analysis optimization
        self.analysis_frame_skip = 3  # Analyze every 3rd frame (8 FPS effective analysis)
        
    def get_center(self, rect):
        """Get center point of rectangle"""
        x, y, w, h = rect
        return (x + w // 2, y + h // 2)
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        if point1 is None or point2 is None:
            return 0
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_eye_in_face(self, face_roi):
        """Detect eyes within a face region - optimized for performance"""
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, 
            scaleFactor=1.15,  # Slightly larger steps for performance
            minNeighbors=4,    # Reduced for faster detection
            minSize=(12, 12),  # Slightly larger minimum
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return eyes
    
    def analyze_frame(self, frame):
        """Analyze a single frame for face and eye detection - optimized"""
        # Resize frame for faster processing (optional)
        height, width = frame.shape[:2]
        if width > 640:
            scale_factor = 640 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        else:
            scale_factor = 1.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optimize face detection parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(60, 60),  # Larger minimum for performance
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        analysis_result = {
            'faces_detected': len(faces),
            'face_movement': 0,
            'eye_movement': 0,
            'looking_away': False,
            'eyes_detected': 0,
            'frame_scale': scale_factor
        }
        
        if len(faces) > 0:
            # Use the largest face (most likely the main subject)
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            
            # Adjust coordinates back to original scale
            if scale_factor != 1.0:
                x = int(x / scale_factor)
                y = int(y / scale_factor)
                w = int(w / scale_factor)
                h = int(h / scale_factor)
            
            # Calculate face center
            face_center = self.get_center((x, y, w, h))
            
            # Calculate face movement
            if self.prev_face_center is not None:
                face_movement = self.calculate_distance(face_center, self.prev_face_center)
                analysis_result['face_movement'] = face_movement
                
                # Detect if face moved significantly (potential looking away)
                if face_movement > self.face_movement_threshold:
                    analysis_result['looking_away'] = True
            
            self.prev_face_center = face_center
            
            # Detect eyes within the face region
            face_roi_y1 = max(0, int(y * scale_factor))
            face_roi_y2 = min(gray.shape[0], int((y + h) * scale_factor))
            face_roi_x1 = max(0, int(x * scale_factor))
            face_roi_x2 = min(gray.shape[1], int((x + w) * scale_factor))
            
            face_roi = gray[face_roi_y1:face_roi_y2, face_roi_x1:face_roi_x2]
            
            if face_roi.size > 0:
                eyes = self.detect_eye_in_face(face_roi)
                analysis_result['eyes_detected'] = len(eyes)
                
                # Analyze eye movement if eyes are detected
                if len(eyes) >= 2:
                    # Sort eyes by x-coordinate to get left and right eye
                    eyes = sorted(eyes, key=lambda e: e[0])
                    
                    # Take the two most prominent eyes
                    eye_centers = []
                    for eye in eyes[:2]:
                        ex, ey, ew, eh = eye
                        # Convert eye coordinates back to full frame coordinates
                        eye_center_x = face_roi_x1 + ex + ew//2
                        eye_center_y = face_roi_y1 + ey + eh//2
                        
                        if scale_factor != 1.0:
                            eye_center_x = int(eye_center_x / scale_factor)
                            eye_center_y = int(eye_center_y / scale_factor)
                        
                        eye_centers.append((eye_center_x, eye_center_y))
                    
                    # Calculate eye movement
                    if self.prev_eye_centers is not None and len(self.prev_eye_centers) == len(eye_centers):
                        total_eye_movement = 0
                        for i, (current, previous) in enumerate(zip(eye_centers, self.prev_eye_centers)):
                            movement = self.calculate_distance(current, previous)
                            total_eye_movement += movement
                        
                        avg_eye_movement = total_eye_movement / len(eye_centers)
                        analysis_result['eye_movement'] = avg_eye_movement
                        
                        # Detect significant eye movement
                        if avg_eye_movement > self.eye_movement_threshold:
                            analysis_result['looking_away'] = True
                    
                    self.prev_eye_centers = eye_centers
                else:
                    # If eyes are not detected, consider it as looking away
                    analysis_result['looking_away'] = True
                    self.prev_eye_centers = None
        else:
            # No face detected - definitely looking away
            analysis_result['looking_away'] = True
            self.prev_face_center = None
            self.prev_eye_centers = None
        
        return analysis_result
    
    def analyze_video_for_cheating(self, video_path, output_dir):
        """Analyze 24 FPS video for eye movement and detect potential cheating"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            # Video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Tracking variables
            frame_count = 0
            looking_away_frames = 0
            total_analyzed_frames = 0
            no_face_frames = 0
            
            suspicious_movements = []
            detailed_analysis = []
            
            print(f"Starting optimized video analysis: {total_frames} frames at {fps:.1f} FPS, {duration:.2f} seconds")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Optimized frame analysis - analyze every 3rd frame for 24 FPS (effective 8 FPS analysis)
                if frame_count % self.analysis_frame_skip != 0:
                    continue
                
                total_analyzed_frames += 1
                
                # Analyze current frame
                frame_analysis = self.analyze_frame(frame)
                timestamp = frame_count / fps if fps > 0 else frame_count
                
                # Record detailed analysis (sample for efficiency)
                frame_analysis['timestamp'] = timestamp
                frame_analysis['frame_number'] = frame_count
                
                # Only store every 10th analysis for memory efficiency
                if total_analyzed_frames % 10 == 0:
                    detailed_analysis.append(frame_analysis)
                
                # Count suspicious behavior
                if frame_analysis['looking_away']:
                    looking_away_frames += 1
                    
                    # Record significant movements
                    if (frame_analysis['face_movement'] > self.face_movement_threshold or 
                        frame_analysis['eye_movement'] > self.eye_movement_threshold or
                        frame_analysis['faces_detected'] == 0):
                        
                        movement_data = {
                            'timestamp': timestamp,
                            'face_movement': frame_analysis['face_movement'],
                            'eye_movement': frame_analysis['eye_movement'],
                            'faces_detected': frame_analysis['faces_detected'],
                            'eyes_detected': frame_analysis['eyes_detected'],
                            'type': 'suspicious_behavior'
                        }
                        suspicious_movements.append(movement_data)
                
                if frame_analysis['faces_detected'] == 0:
                    no_face_frames += 1
                
                # Progress indicator (less frequent for performance)
                if total_analyzed_frames % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Analysis progress: {progress:.1f}% ({total_analyzed_frames} frames analyzed)")
            
            cap.release()
            
            # Calculate metrics
            if total_analyzed_frames > 0:
                looking_away_percentage = (looking_away_frames / total_analyzed_frames) * 100
                no_face_percentage = (no_face_frames / total_analyzed_frames) * 100
                
                # Enhanced scoring logic for 24 FPS
                base_cheating_score = looking_away_percentage * 0.7  # Base score from looking away
                no_face_penalty = no_face_percentage * 0.3  # Additional penalty for no face detection
                movement_penalty = min(25, len(suspicious_movements) * 2.5)  # Adjusted penalty for lower frame rate
                
                cheating_score = min(100, base_cheating_score + no_face_penalty + movement_penalty)
                
                # Determine if cheating is detected (adjusted threshold for 24 FPS)
                is_cheating = cheating_score > 30  # Slightly higher threshold for 24 FPS
                
                # Generate analysis report
                analysis_result = {
                    'video_duration': duration,
                    'video_fps': fps,
                    'analysis_fps': fps / self.analysis_frame_skip,
                    'total_frames': total_frames,
                    'total_frames_analyzed': total_analyzed_frames,
                    'looking_away_frames': looking_away_frames,
                    'no_face_frames': no_face_frames,
                    'looking_away_percentage': looking_away_percentage,
                    'no_face_percentage': no_face_percentage,
                    'cheating_score': cheating_score,
                    'is_cheating_detected': is_cheating,
                    'suspicious_movements': suspicious_movements[:20],  # Top 20 movements
                    'total_suspicious_movements': len(suspicious_movements),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'analysis_method': 'OpenCV Haar Cascades (24 FPS Optimized)',
                    'optimization_notes': f'Frame skip: {self.analysis_frame_skip}, Effective analysis rate: {fps/self.analysis_frame_skip:.1f} FPS'
                }
                
                # Save optimized detailed analysis
                detailed_analysis_file = os.path.join(output_dir, 'detailed_eye_analysis.json')
                with open(detailed_analysis_file, 'w') as f:
                    json.dump({
                        'summary': analysis_result,
                        'frame_samples': detailed_analysis[-50:]  # Last 50 samples for debugging
                    }, f, indent=2)
                
                # Save analysis results
                analysis_file = os.path.join(output_dir, 'eye_analysis.json')
                with open(analysis_file, 'w') as f:
                    json.dump(analysis_result, f, indent=2)
                
                # Create human-readable report
                report_file = os.path.join(output_dir, 'cheating_analysis.txt')
                with open(report_file, 'w') as f:
                    f.write("OPTIMIZED OPENCV EYE TRACKING ANALYSIS REPORT\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Video Duration: {duration:.2f} seconds\n")
                    f.write(f"Video FPS: {fps:.2f}\n")
                    f.write(f"Analysis FPS: {fps/self.analysis_frame_skip:.2f} (optimized)\n")
                    f.write(f"Total Frames: {total_frames}\n")
                    f.write(f"Frames Analyzed: {total_analyzed_frames}\n")
                    f.write(f"Analysis Method: OpenCV Haar Cascades (24 FPS Optimized)\n\n")
                    
                    f.write("BEHAVIORAL ANALYSIS:\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Looking Away Percentage: {looking_away_percentage:.2f}%\n")
                    f.write(f"No Face Detection: {no_face_percentage:.2f}%\n")
                    f.write(f"Suspicious Movements: {len(suspicious_movements)}\n")
                    f.write(f"Cheating Score: {cheating_score:.2f}/100\n")
                    f.write(f"Cheating Detected: {'YES' if is_cheating else 'NO'}\n\n")
                    
                    f.write("OPTIMIZATION DETAILS:\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Frame Skip Rate: {self.analysis_frame_skip}\n")
                    f.write(f"Effective Analysis Rate: {fps/self.analysis_frame_skip:.1f} FPS\n")
                    f.write(f"Performance Gain: ~{self.analysis_frame_skip}x faster processing\n\n")
                    
                    if suspicious_movements:
                        f.write("TOP SUSPICIOUS MOMENTS:\n")
                        f.write("-" * 25 + "\n")
                        for i, movement in enumerate(suspicious_movements[:10]):
                            f.write(f"{i+1}. Time: {movement['timestamp']:.2f}s - ")
                            if movement['faces_detected'] == 0:
                                f.write("Face not detected\n")
                            else:
                                f.write(f"Movement detected (Face: {movement['face_movement']:.1f}px, Eyes: {movement['eye_movement']:.1f}px)\n")
                    
                    f.write(f"\nANALYSIS COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                print(f"Optimized analysis completed: Score {cheating_score:.2f}, Cheating: {is_cheating}")
                return analysis_result
            else:
                raise Exception("No frames could be analyzed")
                
        except Exception as e:
            print(f"Error in optimized eye tracking analysis: {e}")
            return {
                'error': str(e),
                'is_cheating_detected': False,
                'cheating_score': 0,
                'video_duration': 0,
                'total_frames_analyzed': 0,
                'analysis_timestamp': datetime.now().isoformat()
            }

def process_interview_async(session_id, username, video_path, audio_path=None):
    """Process interview in background thread - optimized for separate streams"""
    try:
        print(f"Starting optimized background processing for session: {session_id}")
        user_dir = os.path.join(USERS_FOLDER, username)
        session_dir = os.path.join(user_dir, 'interview', session_id)
        
        # Initialize processing status
        processing_status = {
            'video_analysis_completed': False,
            'audio_ready': False,
            'total_processing_completed': False
        }
        
        # 1. Check if audio is already available (separate recording)
        audio_ready = False
        if audio_path and os.path.exists(audio_path):
            print("Audio stream already available - no extraction needed")
            audio_ready = True
        else:
            # If no separate audio, we'll skip audio processing for now
            print("No separate audio stream provided - focusing on video analysis")
            audio_ready = True  # Mark as ready since we're not processing audio
        
        processing_status['audio_ready'] = audio_ready
        
        # 2. Analyze video for eye movement (optimized for 24 FPS)
        print("Starting optimized eye movement analysis...")
        eye_tracker = OptimizedEyeTracker()
        analysis_result = eye_tracker.analyze_video_for_cheating(video_path, session_dir)
        
        processing_status['video_analysis_completed'] = 'error' not in analysis_result
        
        # 3. Create final analysis file in user directory
        final_analysis_path = os.path.join(user_dir, 'interview_analysis.json')
        final_analysis = {
            'session_id': session_id,
            'processing_completed': True,
            'processing_timestamp': datetime.now().isoformat(),
            'cheating_analysis': analysis_result,
            'audio_extracted': audio_ready,
            'video_analyzed': processing_status['video_analysis_completed'],
            'analysis_method': 'OpenCV Haar Cascades (24 FPS Optimized)',
            'optimization_features': [
                '24 FPS recording',
                'Separate audio/video streams',
                'Optimized frame analysis',
                'Reduced memory usage',
                'Performance-tuned thresholds'
            ]
        }
        
        with open(final_analysis_path, 'w') as f:
            json.dump(final_analysis, f, indent=2)
        
        print(f"Optimized interview processing completed for session: {session_id}")
        
    except Exception as e:
        print(f"Error in optimized processing: {e}")
        # Create error file
        user_dir = os.path.join(USERS_FOLDER, username)
        error_analysis_path = os.path.join(user_dir, 'interview_analysis.json')
        error_analysis = {
            'session_id': session_id,
            'processing_completed': False,
            'error': str(e),
            'processing_timestamp': datetime.now().isoformat(),
            'analysis_method': 'OpenCV Haar Cascades (24 FPS Optimized)'
        }
        
        with open(error_analysis_path, 'w') as f:
            json.dump(error_analysis, f, indent=2)

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
        self.audio_filename = None  # New field for separate audio
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
            'audio_filename': self.audio_filename,
            'completed': self.completed,
            'duration_seconds': self.duration_seconds,
            'duration_formatted': self.format_duration(),
            'role_applied': self.role_applied,
            'recording_fps': 24  # Document the optimized FPS
        }
    
    def format_duration(self):
        if self.duration_seconds == 0:
            return "00:00"
        minutes = self.duration_seconds // 60
        seconds = self.duration_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

# Store active sessions
active_sessions = {}

# [Keep all the existing route handlers - they remain the same]
# @app.route('/') through @app.route('/api/user-profile') are unchanged

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

@app.route('/results')
def results_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return send_file('results.html')

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
            session.clear()  # Clear any existing session data
            session['username'] = username
            session.permanent = True
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
    try:
        session.clear()  # Clear all session data
        return jsonify({'success': True, 'message': 'Logged out successfully'})
    except Exception as e:
        print(f"Logout error: {e}")
        return jsonify({'success': True, 'message': 'Logged out'})  # Always succeed for logout

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
        
        print(f"Started optimized interview session: {session_id} for user: {username}")
        return jsonify({'status': 'success', 'sessionId': session_id, 'recording_fps': 24})
        
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
    """Save optimized 24 FPS video recording and optional separate audio"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        session_id = request.form.get('sessionId')
        video_file = request.files.get('video')
        audio_file = request.files.get('audio')  # Optional separate audio
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
        
        # Save video file (24 FPS optimized)
        session_dir = os.path.join(USERS_FOLDER, username, 'interview', session_id)
        video_filename = f"interview_{session_id}_24fps.webm"
        video_filepath = os.path.join(session_dir, video_filename)
        
        print(f"Saving optimized 24 FPS video to: {video_filepath}")
        video_file.save(video_filepath)
        
        # Save separate audio file if provided
        audio_filepath = None
        if audio_file:
            audio_filename = f"interview_{session_id}_audio.wav"
            audio_filepath = os.path.join(session_dir, audio_filename)
            print(f"Saving separate audio to: {audio_filepath}")
            audio_file.save(audio_filepath)
            interview_session.audio_filename = audio_filename
        
        # Update session
        interview_session.recording_filename = video_filename
        interview_session.questions_answered = len(interview_session.question_timings)
        
        # Save updated session info
        with open(os.path.join(session_dir, 'session_info.json'), 'w') as f:
            json.dump(interview_session.to_dict(), f, indent=2)
        
        # Start optimized background processing
        threading.Thread(
            target=process_interview_async, 
            args=(session_id, username, video_filepath, audio_filepath),
            daemon=True
        ).start()
        
        print(f"Optimized recording saved successfully: Video={video_filename}, Audio={interview_session.audio_filename}")
        return jsonify({
            'status': 'success', 
            'video_filename': video_filename,
            'audio_filename': interview_session.audio_filename,
            'fps': 24
        })
        
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
        
        print(f"Optimized interview completed: {session_id} for user: {username}")
        return jsonify({'status': 'success', 'message': 'Interview completed successfully', 'redirect': '/results'})
        
    except Exception as e:
        print(f"Finish interview error: {e}")
        return jsonify({'error': 'Failed to finish interview'}), 500

@app.route('/api/processing-status')
def processing_status():
    """Check if interview processing is complete"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        username = session['username']
        user_dir = os.path.join(USERS_FOLDER, username)
        analysis_file = os.path.join(user_dir, 'interview_analysis.json')
        
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            return jsonify({
                'processing_completed': analysis.get('processing_completed', False),
                'analysis': analysis
            })
        else:
            return jsonify({'processing_completed': False})
            
    except Exception as e:
        print(f"Processing status error: {e}")
        return jsonify({'processing_completed': False, 'error': str(e)})

@app.route('/api/interview-results')
def interview_results():
    """Get interview analysis results"""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        username = session['username']
        user_dir = os.path.join(USERS_FOLDER, username)
        analysis_file = os.path.join(user_dir, 'interview_analysis.json')
        
        if not os.path.exists(analysis_file):
            return jsonify({'error': 'No analysis results found'}), 404
        
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        # Get latest session info
        sessions = []
        interview_dir = os.path.join(user_dir, 'interview')
        if os.path.exists(interview_dir):
            for session_id in os.listdir(interview_dir):
                session_dir = os.path.join(interview_dir, session_id)
                session_info_path = os.path.join(session_dir, 'session_info.json')
                
                if os.path.exists(session_info_path):
                    with open(session_info_path, 'r') as f:
                        session_data = json.load(f)
                    sessions.append(session_data)
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        latest_session = sessions[0] if sessions else None
        
        return jsonify({
            'analysis': analysis,
            'latest_session': latest_session,
            'total_sessions': len(sessions)
        })
        
    except Exception as e:
        print(f"Interview results error: {e}")
        return jsonify({'error': 'Failed to get results'}), 500

def generate_session_report(session_id, interview_session, username):
    """Generate a comprehensive report for the interview session"""
    session_dir = os.path.join(USERS_FOLDER, username, 'interview', session_id)
    report_path = os.path.join(session_dir, 'interview_report.txt')
    
    try:
        user_info = get_user_info(username)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("AI INTERVIEWER - OPTIMIZED SESSION REPORT\n")
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
                f.write(f"Video File: {interview_session.recording_filename}\n")
            if interview_session.audio_filename:
                f.write(f"Audio File: {interview_session.audio_filename}\n")
            
            f.write(f"Status: {'Completed' if interview_session.completed else 'Incomplete'}\n\n")
            
            f.write("OPTIMIZATION FEATURES:\n")
            f.write("-" * 30 + "\n")
            f.write("- 24 FPS optimized recording\n")
            f.write("- Separate audio/video streams\n")
            f.write("- 3x faster video analysis\n")
            f.write("- Reduced memory usage\n")
            f.write("- Performance-tuned thresholds\n\n")
            
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
            
            f.write("\nTECHNICAL DETAILS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"- System: AI Interviewer Web Platform (Optimized)\n")
            f.write(f"- Recording FPS: 24\n")
            f.write(f"- Analysis Method: OpenCV Haar Cascades (24 FPS Optimized)\n")
            f.write(f"- Session Directory: {session_dir}\n")
            
        print(f"Optimized report generated: {report_path}")
        
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
                        
                        # Add file size information for both video and audio
                        recording_file = session_data.get('recording_filename')
                        audio_file = session_data.get('audio_filename')
                        
                        total_size_mb = 0
                        if recording_file:
                            recording_path = os.path.join(session_dir, recording_file)
                            if os.path.exists(recording_path):
                                video_size_mb = os.path.getsize(recording_path) / (1024 * 1024)
                                session_data['video_size_mb'] = round(video_size_mb, 2)
                                total_size_mb += video_size_mb
                        
                        if audio_file:
                            audio_path = os.path.join(session_dir, audio_file)
                            if os.path.exists(audio_path):
                                audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                                session_data['audio_size_mb'] = round(audio_size_mb, 2)
                                total_size_mb += audio_size_mb
                        
                        session_data['total_size_mb'] = round(total_size_mb, 2)
                        sessions.append(session_data)
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return jsonify(sessions)
        
    except Exception as e:
        print(f"List sessions error: {e}")
        return jsonify({'error': 'Failed to list sessions'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("AI INTERVIEWER - OPTIMIZED 24 FPS WEB SERVER")
    print("=" * 60)
    print(f"🚀 Starting optimized server...")
    print(f"🌐 Access URL: http://localhost:5000")
    print(f"📁 Base directory: {os.path.abspath(BASE_DIR)}")
    print(f"👥 Users folder: {os.path.abspath(USERS_FOLDER)}")
    print(f"🔐 Login details: {os.path.abspath(LOGIN_DETAILS_CSV)}")
    print(f"🎥 Recording FPS: 24 (Optimized)")
    print(f"🔍 Analysis method: OpenCV Haar Cascades (24 FPS Optimized)")
    print(f"⚡ Performance: 3x faster analysis, reduced memory usage")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)