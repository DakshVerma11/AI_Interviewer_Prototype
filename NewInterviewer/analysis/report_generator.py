# analysis/report_generator.py
import os
import json
import random
from datetime import datetime

class InterviewResultGenerator:
    """Generates comprehensive interview results by aggregating multiple data sources."""
    
    def __init__(self, username, session_id, base_dir):
        self.username = username
        self.session_id = session_id
        self.base_dir = base_dir
        self.user_dir = os.path.join(base_dir, 'Users', username)
        self.session_dir = os.path.join(self.user_dir, 'interview', session_id)
        
    def get_session_info(self):
        """Retrieves basic session information from session_info.json."""
        session_info_path = os.path.join(self.session_dir, 'session_info.json')
        if os.path.exists(session_info_path):
            with open(session_info_path, 'r') as f:
                return json.load(f)
        return {}
        
    def get_eye_analysis(self):
        """Retrieves eye tracking/cheating analysis data from eye_analysis.json."""
        eye_analysis_path = os.path.join(self.session_dir, 'eye_analysis.json')
        if os.path.exists(eye_analysis_path):
            with open(eye_analysis_path, 'r') as f:
                return json.load(f)
        return {}
        
    def get_audio_analysis(self):
        """Retrieves audio analysis data from audio_analysis.json."""
        audio_analysis_path = os.path.join(self.session_dir, 'audio_analysis.json')
        if os.path.exists(audio_analysis_path):
            with open(audio_analysis_path, 'r') as f:
                return json.load(f)
        return {}
        
    def get_question_performance(self, session_info):
        """
        Generates performance metrics for individual questions.
        Currently uses random scores as placeholders, to be replaced by actual NLP analysis.
        """
        question_performance = []
        
        if not session_info or 'question_timings' not in session_info:
            return []
            
        timings = session_info.get('question_timings', [])
        
        for idx, timing in enumerate(timings):
            # Placeholder: Generate random scores for relevance, confidence, clarity
            # In a real application, these would come from NLP models analyzing the answer content
            relevance = random.uniform(0.5, 1.0) if idx % 2 == 0 else random.uniform(0.3, 0.9)
            confidence = random.uniform(0.4, 0.95)
            clarity = random.uniform(0.6, 0.98)
            
            question_data = {
                'question_index': timing.get('questionIndex', idx),
                'question': timing.get('question', f'Question {idx+1}'),
                'time_from_start': timing.get('timeFromStart', 0),
                'duration': random.randint(15, 120),  # Random duration for placeholder
                'scores': {
                    'relevance': round(relevance, 2),
                    'confidence': round(confidence, 2),
                    'clarity': round(clarity, 2),
                    'overall': round((relevance + confidence + clarity) / 3, 2)
                }
            }
            question_performance.append(question_data)
            
        return question_performance
        
    def generate_overall_scores(self, eye_analysis, audio_analysis, question_performance):
        """
        Generates overall interview scores based on integrity, content, delivery, and vocal metrics.
        """
        # Default values in case data is missing
        cheating_score = eye_analysis.get('cheating_score', 0)
        is_cheating = eye_analysis.get('is_cheating_detected', False)
        
        audio_metrics = audio_analysis.get('metrics', {}) if audio_analysis else {}
            
        # Calculate average scores from question performance
        avg_relevance = 0
        avg_confidence = 0
        avg_clarity = 0
        
        if question_performance:
            total_questions = len(question_performance)
            avg_relevance = sum(q['scores']['relevance'] for q in question_performance) / total_questions
            avg_confidence = sum(q['scores']['confidence'] for q in question_performance) / total_questions
            avg_clarity = sum(q['scores']['clarity'] for q in question_performance) / total_questions
        
        # Calculate interview integrity score (inversely related to cheating score)
        integrity_score = max(0, 100 - cheating_score)
        
        # Calculate content score (based on average relevance)
        content_score = avg_relevance * 100
        
        # Calculate delivery score (based on average confidence and clarity)
        delivery_score = ((avg_confidence + avg_clarity) / 2) * 100
        
        # Vocal confidence from audio analysis
        vocal_confidence = audio_metrics.get('vocal_confidence', random.uniform(0.6, 0.9)) * 100
        
        # Calculate overall interview score with weighted averages
        overall_score = (
            (integrity_score * 0.3) +    # 30% weight to integrity
            (content_score * 0.4) +      # 40% weight to content relevance
            (delivery_score * 0.2) +     # 20% weight to delivery
            (vocal_confidence * 0.1)     # 10% weight to vocal confidence
        )
        
        return {
            'integrity_score': round(integrity_score, 1),
            'content_score': round(content_score, 1),
            'delivery_score': round(delivery_score, 1),
            'vocal_confidence': round(vocal_confidence, 1),
            'overall_score': round(overall_score, 1),
            'is_cheating_detected': is_cheating
        }
    
    def generate_feedback(self, scores, audio_metrics, eye_analysis):
        """Generates personalized textual feedback based on various performance scores."""
        feedback = []
        
        # Integrity feedback
        if scores['is_cheating_detected']:
            feedback.append({
                'category': 'Integrity',
                'type': 'negative',
                'message': 'Suspicious eye movements were detected during your interview. Maintaining eye contact shows confidence and honesty.'
            })
        else:
            feedback.append({
                'category': 'Integrity',
                'type': 'positive',
                'message': 'You maintained good eye contact throughout the interview, which demonstrates confidence and integrity.'
            })
        
        # Content feedback
        if scores['content_score'] >= 80:
            feedback.append({
                'category': 'Content',
                'type': 'positive',
                'message': 'Your answers were highly relevant to the questions asked, showing good understanding of the requirements.'
            })
        elif scores['content_score'] >= 60:
            feedback.append({
                'category': 'Content',
                'type': 'neutral',
                'message': 'Your answers were mostly relevant, but could be more focused on addressing the specific questions asked.'
            })
        else:
            feedback.append({
                'category': 'Content',
                'type': 'negative',
                'message': 'Your answers often drifted from the questions. Try to stay more focused on addressing what was specifically asked.'
            })
        
        # Delivery feedback
        if scores['delivery_score'] >= 80:
            feedback.append({
                'category': 'Delivery',
                'type': 'positive',
                'message': 'Your delivery was clear and confident, making your points easy to understand.'
            })
        elif scores['delivery_score'] >= 60:
            feedback.append({
                'category': 'Delivery',
                'type': 'neutral',
                'message': 'Your delivery was adequate but could be improved with more clarity and confidence.'
            })
        else:
            feedback.append({
                'category': 'Delivery',
                'type': 'negative',
                'message': 'Your delivery lacked clarity. Consider practicing speaking more clearly and confidently.'
            })
        
        # Filler words feedback
        if audio_metrics and 'fillers' in audio_metrics:
            fillers = audio_metrics['fillers']
            total_fillers = sum(fillers.values()) if fillers else 0
            
            if total_fillers > 10:
                feedback.append({
                    'category': 'Speech',
                    'type': 'negative',
                    'message': f'You used filler words (like "um", "uh", "like") {total_fillers} times. Reducing these will make your answers sound more confident.'
                })
            elif total_fillers > 5:
                feedback.append({
                    'category': 'Speech',
                    'type': 'neutral',
                    'message': f'You occasionally used filler words ({total_fillers} instances). Being more conscious of these can improve your delivery.'
                })
            else:
                feedback.append({
                    'category': 'Speech',
                    'type': 'positive',
                    'message': 'You used very few filler words, which made your speech sound professional and prepared.'
                })
        
        # Speaking rate feedback
        if audio_metrics and 'rate_wpm' in audio_metrics:
            rate = audio_metrics['rate_wpm']
            
            if rate < 120:
                feedback.append({
                    'category': 'Pace',
                    'type': 'neutral',
                    'message': f'Your speaking pace ({rate} words per minute) was somewhat slow. A slightly faster pace might keep the interviewer more engaged.'
                })
            elif rate > 180:
                feedback.append({
                    'category': 'Pace',
                    'type': 'neutral',
                    'message': f'Your speaking pace ({rate} words per minute) was quite fast. Slowing down slightly might help clarity.'
                })
            else:
                feedback.append({
                    'category': 'Pace',
                    'type': 'positive',
                    'message': f'Your speaking pace ({rate} words per minute) was excellent - neither too fast nor too slow.'
                })
        
        return feedback
        
    def generate_final_results(self):
        """
        Generates and saves the comprehensive final interview results by
        aggregating all available analysis data.
        """
        # Get all necessary data from respective JSON files
        session_info = self.get_session_info()
        eye_analysis = self.get_eye_analysis()
        audio_analysis = self.get_audio_analysis()
        
        # Generate question performance data (currently with placeholders)
        question_performance = self.get_question_performance(session_info)
        
        # Extract audio metrics
        audio_metrics = audio_analysis.get('metrics', {}) if audio_analysis else {}
        
        # Generate overall scores
        overall_scores = self.generate_overall_scores(eye_analysis, audio_analysis, question_performance)
        
        # Generate personalized feedback
        feedback = self.generate_feedback(overall_scores, audio_metrics, eye_analysis)
        
        # Compile all data into the final results dictionary
        final_results = {
            'session_id': self.session_id,
            'username': self.username,
            'interview_date': session_info.get('start_time', datetime.now().isoformat()),
            'role_applied': session_info.get('role_applied', 'Not specified'),
            'duration': session_info.get('duration_formatted', '00:00'),
            'questions_total': session_info.get('total_questions', 0),
            'questions_answered': session_info.get('questions_answered', 0),
            'overall_scores': overall_scores,
            'question_performance': question_performance,
            'feedback': feedback,
            'cheating_analysis': eye_analysis,
            'audio_analysis': audio_analysis,
            'generation_timestamp': datetime.now().isoformat(),
            'results_version': '1.0'
        }
        
        # Save final results to the session directory
        results_path = os.path.join(self.session_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
            
        # Also save a copy to the user's root directory for easier access to the latest results
        user_results_path = os.path.join(self.user_dir, 'latest_results.json')
        with open(user_results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
            
        return final_results

