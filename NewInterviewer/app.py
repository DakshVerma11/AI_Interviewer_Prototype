# app.py (Main Flask Application File)
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import os
import json
import csv
import threading
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename

# Import configurations
from config import BASE_DIR, LOGIN_DETAILS_CSV, USERS_FOLDER, HF_TOKEN

# Import modularized components
from utils.user_manager import hash_password, verify_password, create_user_folder, \
                               check_user_exists, authenticate_user, get_user_info
from utils.session_manager import InterviewSession
from analysis.eye_tracking import OptimizedEyeTracker
from analysis.audio_processing import extract_audio_ffmpeg, run_audio_processing_async
from analysis.report_generator import InterviewResultGenerator

app = Flask(__name__)
app.secret_key = 'ai_interviewer_secret_key_2025'  # IMPORTANT: Change this in production

# Store active sessions in memory (for simplicity; could use a database for persistence)
active_sessions = {}

# --- Route Definitions ---

@app.route('/')
def index():
    """Redirects to dashboard if logged in, otherwise to login page."""
    if 'username' in session:
        return send_file('dashboard.html')
    return send_file('login.html')

@app.route('/login')
def login_page():
    """Serves the login HTML page."""
    return send_file('login.html')

@app.route('/register')
def register_page():
    """Serves the registration HTML page."""
    return send_file('register.html')

@app.route('/dashboard')
def dashboard():
    """Serves the dashboard HTML page, requires login."""
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return send_file('dashboard.html')

@app.route('/interview')
def interview_page():
    """Serves the interview HTML page, requires login."""
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return send_file('interview.html')

@app.route('/results')
def results_page():
    """Serves the results HTML page, requires login."""
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return send_file('results.html')

# --- API Endpoints ---

@app.route('/api/register', methods=['POST'])
def register():
    """API endpoint for user registration."""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        # Input validation
        if not all([username, email, password]):
            return jsonify({'error': 'All fields are required'}), 400
        
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        if '@' not in email:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if user already exists using user_manager
        if check_user_exists(username, email):
            return jsonify({'error': 'Username or email already exists'}), 400
        
        # Hash password and save user details
        hashed_password = hash_password(password)
        
        with open(LOGIN_DETAILS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([username, email, hashed_password, datetime.now().isoformat()])
        
        # Create user-specific folder structure
        create_user_folder(username)
        
        print(f"User registered: {username}")
        return jsonify({'success': True, 'message': 'Registration successful'})
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """API endpoint for user login."""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
        # Authenticate user using user_manager
        if authenticate_user(username, password):
            session.clear()  # Clear any existing session data
            session['username'] = username
            session.permanent = True # Make session permanent
            user_info = get_user_info(username)
            print(f"User logged in: {username}")
            return jsonify({'success': True, 'user': user_info})
        else:
            return jsonify({'error': 'Invalid username or password'}), 401
            
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """API endpoint for user logout."""
    try:
        username = session.pop('username', None) # Remove username from session
        session.clear()  # Clear all session data
        print(f"User logged out: {username}")
        return jsonify({'success': True, 'message': 'Logged out successfully'})
    except Exception as e:
        print(f"Logout error: {e}")
        # Even if an error occurs, try to clear session and redirect
        session.clear()
        return jsonify({'success': True, 'message': 'Logged out (with potential error)'})

@app.route('/api/current-user')
def current_user():
    """API endpoint to get current logged-in user information."""
    if 'username' in session:
        user_info = get_user_info(session['username'])
        return jsonify({'user': user_info})
    return jsonify({'user': None})

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    """API endpoint to upload user resume and set role."""
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
        
        # Save resume to user's directory
        user_dir = os.path.join(USERS_FOLDER, username)
        filename = secure_filename(f"resume_{username}.pdf")
        filepath = os.path.join(user_dir, filename)
        file.save(filepath)
        
        # Save role information to profile.json
        role_info = {
            'role': role,
            'resume_filename': filename,
            'upload_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(user_dir, 'profile.json'), 'w') as f:
            json.dump(role_info, f, indent=2)
        
        print(f"Resume uploaded for {username}, role: {role}")
        return jsonify({'success': True, 'message': 'Resume uploaded successfully'})
        
    except Exception as e:
        print(f"Resume upload error: {e}")
        return jsonify({'error': 'Failed to upload resume'}), 500

@app.route('/api/user-profile')
def user_profile():
    """API endpoint to get user profile information."""
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
    """API endpoint to load interview questions from user-specific CSV."""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        username = session['username']
        user_questions_file = os.path.join(USERS_FOLDER, username, 'questions.csv')
        
        if not os.path.exists(user_questions_file):
            return jsonify({'error': 'INTERVIEW_NOT_READY'}) # Signal to frontend
        
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
    """API endpoint to initialize a new interview session."""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        start_time = data.get('startTime')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        username = session['username']
        
        # Create new InterviewSession object
        interview_session = InterviewSession(session_id, username, start_time)
        active_sessions[session_id] = interview_session # Store in active sessions
        
        # Create session directory
        session_dir = os.path.join(USERS_FOLDER, username, 'interview', session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Load total questions count for the session
        user_questions_file = os.path.join(USERS_FOLDER, username, 'questions.csv')
        try:
            with open(user_questions_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                interview_session.total_questions = len([row for row in reader if row and row[0].strip()])
        except Exception as e:
            print(f"Error loading questions for session: {e}")
            return jsonify({'error': 'Questions file not found or empty'}), 400
        
        # Get role applied for from user profile
        profile_path = os.path.join(USERS_FOLDER, username, 'profile.json')
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile = json.load(f)
                interview_session.role_applied = profile.get('role', '')
        
        # Save initial session info to file
        with open(os.path.join(session_dir, 'session_info.json'), 'w') as f:
            json.dump(interview_session.to_dict(), f, indent=2)
        
        print(f"Started interview session: {session_id} for user: {username}")
        return jsonify({'status': 'success', 'sessionId': session_id, 'recording_fps': 24})
        
    except Exception as e:
        print(f"Start session error: {e}")
        return jsonify({'error': 'Failed to start session'}), 500

@app.route('/api/update-timings', methods=['POST'])
def update_timings():
    """API endpoint to update question timings during the interview."""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        question_timings = data.get('questionTimings', [])
        
        interview_session = active_sessions.get(session_id)
        if not interview_session:
            return jsonify({'error': 'Invalid session'}), 400
        
        # Update session object in memory
        interview_session.question_timings = question_timings
        interview_session.questions_answered = len(question_timings)
        
        # Save updated session info to file
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
    """API endpoint to save optimized video recording and optional separate audio."""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        session_id = request.form.get('sessionId')
        video_file = request.files.get('video')
        audio_file = request.files.get('audio')  # Optional separate audio stream
        question_timings_str = request.form.get('questionTimings', '[]')
        end_time = request.form.get('endTime')
        
        if not all([session_id, video_file]):
            return jsonify({'error': 'Missing required fields (sessionId, video)'}), 400
        
        interview_session = active_sessions.get(session_id)
        if not interview_session:
            return jsonify({'error': 'Invalid session'}), 400
        
        username = session['username']
        session_dir = os.path.join(USERS_FOLDER, username, 'interview', session_id)

        # Parse and update question timings
        try:
            question_timings = json.loads(question_timings_str)
            interview_session.question_timings = question_timings
        except json.JSONDecodeError:
            print("Warning: Could not decode questionTimings JSON.")
            pass # Continue without updating timings if JSON is invalid
        
        # Set end time and calculate duration
        if end_time:
            interview_session.end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            interview_session.duration_seconds = int((interview_session.end_time - interview_session.start_time).total_seconds())
        
        # Save video file (optimized 24 FPS)
        video_filename = f"interview_{session_id}_24fps.webm"
        video_filepath = os.path.join(session_dir, video_filename)
        
        print(f"Saving optimized 24 FPS video to: {video_filepath}")
        video_file.save(video_filepath)
        interview_session.recording_filename = video_filename # Update session object
        
        # Save separate audio file if provided
        audio_filepath = None
        if audio_file:
            audio_filename = f"interview_{session_id}_audio.wav"
            audio_filepath = os.path.join(session_dir, audio_filename)
            print(f"Saving separate audio to: {audio_filepath}")
            audio_file.save(audio_filepath)
            interview_session.audio_filename = audio_filename # Update session object
        
        # Update session object (questions answered)
        interview_session.questions_answered = len(interview_session.question_timings)
        
        # Save updated session info to file
        with open(os.path.join(session_dir, 'session_info.json'), 'w') as f:
            json.dump(interview_session.to_dict(), f, indent=2)
        
        # Start optimized background processing for analysis
        # This will handle both video (eye tracking) and audio analysis
        threading.Thread(
            target=process_interview_async, 
            args=(session_id, username, video_filepath, audio_filepath),
            daemon=True # Daemon thread allows main program to exit without waiting for it
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

def process_interview_async(session_id, username, video_path, audio_path=None):
    """
    Orchestrates the background processing of interview recordings (video and audio analysis).
    This function runs in a separate thread.
    """
    try:
        print(f"Starting optimized background processing for session: {session_id}")
        user_dir = os.path.join(USERS_FOLDER, username)
        session_dir = os.path.join(user_dir, 'interview', session_id)
        
        # Initialize a combined processing status file for the user
        final_analysis_path = os.path.join(user_dir, 'interview_analysis.json')
        
        # Load existing analysis if it exists, otherwise create new
        if os.path.exists(final_analysis_path):
            with open(final_analysis_path, 'r') as f:
                final_analysis = json.load(f)
        else:
            final_analysis = {
                'session_id': session_id,
                'processing_completed': False,
                'processing_timestamp': datetime.now().isoformat(),
                'analysis_method': 'OpenCV Haar Cascades (24 FPS Optimized)',
                'optimization_features': [
                    '24 FPS recording',
                    'Separate audio/video streams',
                    'Optimized frame analysis',
                    'Automated transcription',
                    'Speech metrics analysis'
                ]
            }

        # 1. Extract audio from video if no separate audio stream was provided
        current_audio_path = audio_path
        if not current_audio_path or not os.path.exists(current_audio_path):
            print("No separate audio stream provided or found - extracting from video...")
            current_audio_path = os.path.join(session_dir, f"extracted_audio_{session_id}.wav")
            try:
                extract_audio_ffmpeg(video_path, current_audio_path)
                final_analysis['audio_extracted'] = True
                print(f"Audio extracted to: {current_audio_path}")
            except Exception as e:
                print(f"Error extracting audio: {e}")
                final_analysis['audio_extracted'] = False
                final_analysis['audio_extraction_error'] = str(e)
        else:
            final_analysis['audio_extracted'] = True
        
        # 2. Analyze video for eye movement (OptimizedEyeTracker)
        print("Starting optimized eye movement analysis...")
        eye_tracker = OptimizedEyeTracker()
        eye_analysis_result = eye_tracker.analyze_video_for_cheating(video_path, session_dir)
        final_analysis['cheating_analysis'] = eye_analysis_result
        final_analysis['video_analyzed'] = 'error' not in eye_analysis_result
        
        # 3. Start audio processing pipeline (run_audio_processing_async)
        if final_analysis['audio_extracted'] and os.path.exists(current_audio_path):
            print("Initiating audio processing pipeline...")
            # This function itself starts a new daemon thread and updates status files
            run_audio_processing_async(session_id, username, current_audio_path, BASE_DIR, HF_TOKEN)
            final_analysis['audio_processing_started'] = True
        else:
            final_analysis['audio_processing_started'] = False
            final_analysis['audio_processing_error'] = "Audio not available for processing."
            print("Audio not available for processing (either not extracted or file missing).")

        # Save the updated combined analysis file
        with open(final_analysis_path, 'w') as f:
            json.dump(final_analysis, f, indent=2)
        
        print(f"Optimized interview processing initiated for session: {session_id}")
        
    except Exception as e:
        print(f"Error in optimized processing: {e}")
        # Update combined analysis file with overall error
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

@app.route('/api/finish-interview', methods=['POST'])
def finish_interview():
    """API endpoint to mark interview as completed and trigger final report generation."""
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
        
        # Update final session data in memory
        interview_session.completed = True
        interview_session.question_timings = question_timings
        interview_session.questions_answered = len(question_timings)
        interview_session.duration_seconds = total_duration
        
        if end_time:
            interview_session.end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Save final session info to file
        session_dir = os.path.join(USERS_FOLDER, username, 'interview', session_id)
        with open(os.path.join(session_dir, 'session_info.json'), 'w') as f:
            json.dump(interview_session.to_dict(), f, indent=2)
        
        # Generate comprehensive report (text summary)
        generate_session_report(session_id, interview_session, username)
        
        print(f"Interview completed: {session_id} for user: {username}")
        return jsonify({'status': 'success', 'message': 'Interview completed successfully', 'redirect': '/results'})
        
    except Exception as e:
        print(f"Finish interview error: {e}")
        return jsonify({'error': 'Failed to finish interview'}), 500

def generate_session_report(session_id, interview_session, username):
    """Generates a comprehensive text report and a summary JSON for the interview session."""
    session_dir = os.path.join(USERS_FOLDER, username, 'interview', session_id)
    report_path = os.path.join(session_dir, 'interview_report.txt')
    
    try:
        user_info = get_user_info(username) # Get user info from user_manager
        
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
            
            # Add current timestamp and login info (placeholder for actual user)
            f.write(f"- Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"- Report Generated By: {username}\n")
            
        print(f"Optimized report generated: {report_path}")
        
        # Also generate a final summary JSON file
        summary_path = os.path.join(session_dir, 'final_summary.json')
        summary = {
            'session_id': session_id,
            'username': username,
            'email': user_info.get('email', 'N/A'),
            'role_applied': interview_session.role_applied,
            'interview_date': interview_session.start_time.isoformat(),
            'duration': interview_session.format_duration(),
            'questions_total': interview_session.total_questions,
            'questions_answered': interview_session.questions_answered,
            'completion_rate': round((interview_session.questions_answered/interview_session.total_questions*100), 1) if interview_session.total_questions > 0 else 0,
            'status': 'Completed' if interview_session.completed else 'Incomplete',
            'generation_timestamp': datetime.now().isoformat(),
            'generated_by': username,
            'report_path': report_path
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
    except Exception as e:
        print(f"Error generating report: {e}")

@app.route('/api/processing-status')
def processing_status():
    """
    API endpoint to check if interview processing is complete and trigger
    final results generation if all sub-processes are done.
    """
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        username = session['username']
        user_dir = os.path.join(USERS_FOLDER, username)
        
        # The main analysis status file
        analysis_file = os.path.join(user_dir, 'interview_analysis.json')
        
        # Get the latest session ID for the user
        latest_session_id = None
        interview_dir = os.path.join(user_dir, 'interview')
        if os.path.exists(interview_dir):
            sessions = []
            for s_id in os.listdir(interview_dir):
                session_path = os.path.join(interview_dir, s_id)
                if os.path.isdir(session_path):
                    session_info_path = os.path.join(session_path, 'session_info.json')
                    if os.path.exists(session_info_path):
                        with open(session_info_path, 'r') as f:
                            info = json.load(f)
                        sessions.append({
                            'id': s_id,
                            'start_time': info.get('start_time', '')
                        })
            if sessions:
                sessions.sort(key=lambda x: x['start_time'], reverse=True)
                latest_session_id = sessions[0]['id']

        if not latest_session_id:
            return jsonify({'processing_completed': False, 'message': 'No interview sessions found.'})

        # Check status from the combined analysis file
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            # Check if both video and audio processing are completed
            video_done = analysis.get('video_analyzed', False)
            audio_done = analysis.get('audio_processing_completed', False)
            
            # If both are done, mark overall processing as complete
            if video_done and audio_done and not analysis.get('processing_completed', False):
                analysis['processing_completed'] = True
                analysis['final_results_generated'] = False # Reset for regeneration if needed
                with open(analysis_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                print(f"Overall processing marked complete for session {latest_session_id}")

            # If overall processing is complete but final results haven't been generated yet
            if analysis.get('processing_completed', False) and not analysis.get('final_results_generated', False):
                final_results_path = os.path.join(interview_dir, latest_session_id, 'final_results.json')
                if not os.path.exists(final_results_path):
                    print(f"Generating final results for session {latest_session_id}...")
                    generator = InterviewResultGenerator(username, latest_session_id, BASE_DIR)
                    generator.generate_final_results()
                    
                    # Update analysis to indicate final results are ready
                    analysis['final_results_generated'] = True
                    with open(analysis_file, 'w') as f:
                        json.dump(analysis, f, indent=2)
                    print(f"Final results generated for session {latest_session_id}")
            
            return jsonify({
                'processing_completed': analysis.get('processing_completed', False),
                'final_results_generated': analysis.get('final_results_generated', False),
                'analysis': analysis
            })
        else:
            return jsonify({'processing_completed': False, 'message': 'No analysis file found.'})
            
    except Exception as e:
        print(f"Processing status error: {e}")
        return jsonify({'processing_completed': False, 'error': str(e), 'message': 'Error checking processing status.'})

@app.route('/api/interview-results')
def interview_results():
    """API endpoint to get general interview analysis results (summary from interview_analysis.json)."""
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
        
        # Get latest session info for context on the dashboard
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

@app.route('/api/final-interview-results')
def get_final_interview_results():
    """API endpoint to get comprehensive final interview results (from final_results.json)."""
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        username = session['username']
        
        user_dir = os.path.join(USERS_FOLDER, username)
        interview_dir = os.path.join(user_dir, 'interview')
        
        if not os.path.exists(interview_dir):
            return jsonify({'error': 'No interviews found'}), 404
            
        # Find the most recent session to get its final results
        sessions = []
        for session_id in os.listdir(interview_dir):
            session_path = os.path.join(interview_dir, session_id)
            if os.path.isdir(session_path):
                session_info_path = os.path.join(session_path, 'session_info.json')
                if os.path.exists(session_info_path):
                    with open(session_info_path, 'r') as f:
                        info = json.load(f)
                    sessions.append({
                        'id': session_id,
                        'start_time': info.get('start_time', '')
                    })
        
        if not sessions:
            return jsonify({'error': 'No interview sessions found'}), 404
            
        sessions.sort(key=lambda x: x['start_time'], reverse=True)
        latest_session_id = sessions[0]['id']
        
        # Attempt to load final results; if not found, generate them
        final_results_path = os.path.join(interview_dir, latest_session_id, 'final_results.json')
        
        results = {}
        if os.path.exists(final_results_path):
            with open(final_results_path, 'r') as f:
                results = json.load(f)
        else:
            # If final results don't exist, try to generate them now
            print(f"Final results not found for {latest_session_id}, generating now...")
            generator = InterviewResultGenerator(username, latest_session_id, BASE_DIR)
            results = generator.generate_final_results()
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error getting final results: {e}")
        return jsonify({'error': f'Failed to get results: {str(e)}'}), 500

@app.route('/api/user-sessions')
def list_user_sessions():
    """API endpoint to list all interview sessions for the current user."""
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
                        
                        # Add file size information for both video and audio recordings
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
                        
                        # Check if final results exist for this session
                        final_results_path = os.path.join(session_dir, 'final_results.json')
                        session_data['has_results'] = os.path.exists(final_results_path)
                        
                        sessions.append(session_data)
        
        # Sort sessions by start time (newest first)
        sessions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return jsonify(sessions)
        
    except Exception as e:
        print(f"List sessions error: {e}")
        return jsonify({'error': 'Failed to list sessions'}), 500

# --- Main Application Run ---
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
    print(f"🔄 Results generation: Automatic when processing completes")
    print("=" * 60)
    
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)

