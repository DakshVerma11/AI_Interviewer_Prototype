# utils/session_manager.py
from datetime import datetime

class InterviewSession:
    """Manages interview session data."""
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
        """Converts session data to a dictionary for JSON serialization."""
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
        """Formats duration in seconds to MM:SS string."""
        if self.duration_seconds == 0:
            return "00:00"
        minutes = self.duration_seconds // 60
        seconds = self.duration_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

