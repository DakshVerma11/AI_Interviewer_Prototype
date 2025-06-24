# Integration module to incorporate audio processing into main application
import os
import json
import threading
from datetime import datetime

# Import audio processing functions
from AudioDecoding.analysis_audio import (
    normalize_format, reduce_noise, trim_silence, 
    chunk_audio, aggregate_verbal
)
from AudioDecoding.diarize import diarize_audio, extract_segments, transcribe_speaker_chunks

class AudioProcessor:
    """Audio processing handler for interview recordings"""
    
    def __init__(self, audio_path, output_dir, hf_token):
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.hf_token = hf_token
        self.results = None
        
    def process(self):
        """Process audio file through entire pipeline"""
        try:
            # 1. Preprocessing
            print(f"Starting audio processing for: {self.audio_path}")
            norm_path = self.audio_path.replace('.wav', '_norm.wav')
            normalize_format(self.audio_path, norm_path)
            
            den_path = norm_path.replace('_norm.wav', '_den.wav')
            reduce_noise(norm_path, den_path)
            
            trim_path = den_path.replace('_den.wav', '_trim.wav')
            trim_silence(den_path, trim_path)
            
            # 2. Speaker diarization
            print("Performing speaker diarization...")
            segments = diarize_audio(trim_path, self.hf_token)
            speaker_chunks = extract_segments(trim_path, segments)
            
            # 3. Transcription
            print("Transcribing speaker segments...")
            speaker_transcripts = transcribe_speaker_chunks(speaker_chunks)
            
            # 4. Get metrics
            print("Analyzing speech metrics...")
            chunks_dir = trim_path.replace('.wav', '_chunks')
            chunks = chunk_audio(trim_path, chunks_dir)
            
            import librosa
            y, sr = librosa.load(trim_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Determine question and answer
            sorted_speakers = sorted(speaker_transcripts.keys())
            question, answer = "", ""
            
            if len(sorted_speakers) >= 2:
                question = speaker_transcripts[sorted_speakers[0]]
                answer = speaker_transcripts[sorted_speakers[1]]
            elif len(sorted_speakers) == 1:
                answer = speaker_transcripts[sorted_speakers[0]]
            
            # 5. Get verbal metrics
            verbal_metrics = aggregate_verbal(question, answer, duration, chunks_dir)
            
            # 6. Save results
            self.results = {
                'transcripts': speaker_transcripts,
                'metrics': verbal_metrics,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Save to JSON file
            results_path = os.path.join(self.output_dir, 'audio_analysis.json')
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            print(f"Audio analysis completed and saved to: {results_path}")
            return self.results
            
        except Exception as e:
            error_msg = f"Error in audio processing: {str(e)}"
            print(error_msg)
            
            # Save error information
            error_results = {
                'error': error_msg,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            error_path = os.path.join(self.output_dir, 'audio_analysis_error.json')
            with open(error_path, 'w') as f:
                json.dump(error_results, f, indent=2)
                
            return error_results

def process_audio_async(audio_path, output_dir, hf_token):
    """Process audio in a background thread"""
    processor = AudioProcessor(audio_path, output_dir, hf_token)
    return processor.process()

def run_audio_processing(session_id, username, audio_path, base_dir):
    """Start audio processing in background thread"""
    user_dir = os.path.join(base_dir, 'Users', username)
    session_dir = os.path.join(user_dir, 'interview', session_id)
    
    # Get HF token from environment or config
    hf_token = "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"
    
    # Update processing status
    status_path = os.path.join(session_dir, 'processing_status.json')
    with open(status_path, 'w') as f:
        json.dump({
            'audio_processing': 'in_progress',
            'started_at': datetime.now().isoformat()
        }, f, indent=2)
    
    # Start processing in background
    def process_thread():
        try:
            results = process_audio_async(audio_path, session_dir, hf_token)
            
            # Update combined analysis file
            analysis_path = os.path.join(user_dir, 'interview_analysis.json')
            if os.path.exists(analysis_path):
                with open(analysis_path, 'r') as f:
                    analysis = json.load(f)
                
                analysis['audio_analysis'] = results
                analysis['audio_processing_completed'] = True
                
                with open(analysis_path, 'w') as f:
                    json.dump(analysis, f, indent=2)
            
            # Update processing status
            with open(status_path, 'w') as f:
                json.dump({
                    'audio_processing': 'completed',
                    'completed_at': datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error in audio processing thread: {e}")
            # Update processing status with error
            with open(status_path, 'w') as f:
                json.dump({
                    'audio_processing': 'error',
                    'error': str(e),
                    'error_at': datetime.now().isoformat()
                }, f, indent=2)
    
    # Start thread
    thread = threading.Thread(target=process_thread)
    thread.daemon = True
    thread.start()
    
    return {
        'status': 'started',
        'message': 'Audio processing started in background'
    }