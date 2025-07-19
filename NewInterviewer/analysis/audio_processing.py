# analysis/audio_processing.py
import os
import subprocess
import time
import logging
import tempfile
import whisper
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, effects
from pyannote.audio import Pipeline
import numpy as np
import json
from datetime import datetime

# Ensure numpy float type compatibility
if not hasattr(np, 'float'):
    np.float = float

# --- Audio Utility Functions ---
def extract_audio_ffmpeg(video_path, audio_path):
    """Extracts audio from a video file using FFmpeg."""
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-ac', '1', '-ar', '16000', '-sample_fmt', 's16', audio_path
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

def normalize_format(input_path, output_path):
    """Normalizes audio to 16kHz sample rate and PCM_16 format."""
    y, sr = librosa.load(input_path, sr=16000)
    sf.write(output_path, y, 16000, subtype='PCM_16')

def reduce_noise(wav_path, denoised_path):
    """Reduces noise in an audio file."""
    y, sr = librosa.load(wav_path, sr=16000)
    # Use a short segment from the beginning to estimate noise profile
    noisy_part = y[0:int(sr*0.5)] 
    reduced = nr.reduce_noise(y=y, y_noise=noisy_part, sr=sr)
    sf.write(denoised_path, reduced, sr)

def trim_silence(input_path, trimmed_path):
    """Trims leading and trailing silence from an audio file."""
    audio = AudioSegment.from_wav(input_path)
    trimmed = effects.strip_silence(audio, silence_thresh=-40, chunk_size=10) # Adjusted chunk_size
    trimmed.export(trimmed_path, format='wav')

def chunk_audio(path, out_dir):
    """Chunks an audio file into smaller segments."""
    os.makedirs(out_dir, exist_ok=True)
    audio = AudioSegment.from_wav(path)
    dur_ms = len(audio)
    step_ms = 25000 # 25-second chunks
    chunks = []
    for start in range(0, dur_ms, step_ms):
        end = min(start + 30000, dur_ms) # Max 30-second chunks
        chunk = audio[start:end]
        fn = os.path.join(out_dir, f"chunk_{start}.wav")
        chunk.export(fn, format='wav')
        chunks.append(fn)
        if end == dur_ms: break
    return chunks

# --- Diarization Functions ---
def diarize_audio(audio_path, hf_token):
    """Performs speaker diarization on an audio file."""
    if not hf_token:
        raise ValueError("Missing Hugging Face token for diarization.")

    print(f"[DEBUG] Starting diarization on: {audio_path}")
    try:
        # Load the pre-trained pyannote speaker diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=hf_token
        )
        diarization = pipeline(audio_path)
    except Exception as e:
        raise RuntimeError(f"Diarization failed: {e}")

    segments = []
    # Iterate through diarization results to extract speaker segments
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"[SEGMENT] Speaker: {speaker} | Start: {turn.start:.2f}s | End: {turn.end:.2f}s")
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })
    return segments

def extract_segments(audio_path, segments):
    """Extracts audio segments for each speaker based on diarization results."""
    audio = AudioSegment.from_wav(audio_path)
    speaker_audio = {}
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        speaker = seg["speaker"]
        chunk = audio[start_ms:end_ms]
        # Concatenate chunks for the same speaker
        speaker_audio[speaker] = speaker_audio.get(speaker, AudioSegment.empty()) + chunk
    return speaker_audio

def transcribe_speaker_chunks(speaker_audio, model_size="base"):
    """Transcribes audio chunks for each speaker using Whisper."""
    model = whisper.load_model(model_size) # Load Whisper model
    results = {}

    for speaker, audio in speaker_audio.items():
        # Export audio chunk to a temporary WAV file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            audio.export(temp_path, format="wav")

        print(f"[TRANSCRIBE] Transcribing audio for {speaker}...")
        try:
            # Transcribe the audio using Whisper
            result = model.transcribe(temp_path, fp16=False)
            results[speaker] = result["text"].strip()
        except Exception as e:
            print(f"[ERROR] Whisper failed on {speaker}: {e}")
            results[speaker] = "" # Store empty string on error
        finally:
            os.remove(temp_path) # Clean up temporary file

    return results

# --- NLP Scoring and Verbal Metrics ---
FILLERS = ['uh', 'um', 'like', 'you know', 'so']

def count_fillers(text):
    """Counts occurrences of common filler words in a given text."""
    words = text.lower().split()
    return {f: words.count(f) for f in FILLERS}

def score_relevance(question, answer):
    """
    Calculates a naive lexical overlap score between a question and an answer
    to estimate relevance. (Can be replaced with a more sophisticated NLP model)
    """
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    if not q_words: return 0.0
    return round(len(q_words.intersection(a_words)) / len(q_words), 2)

def score_clarity(text):
    """Estimates clarity based on the presence of filler words."""
    words = text.split()
    filler_count = sum(text.lower().count(f) for f in FILLERS)
    # Clarity decreases with higher filler word ratio
    return round(1 - min(1.0, filler_count / max(1, len(words))), 2)

def estimate_tone_from_features(rate_wpm, filler_ratio):
    """Estimates speaking tone based on speech rate and filler word ratio."""
    if rate_wpm > 160 and filler_ratio < 0.05:
        label = "Confident"
    elif filler_ratio > 0.15:
        label = "Hesitant"
    else:
        label = "Neutral"
    # Confidence is inversely proportional to filler ratio
    confidence = round(min(1.0, max(0.0, 1 - filler_ratio * 2)), 2)
    return {"label": label, "confidence": confidence}

def speech_rate_wpm(text, duration_s):
    """Calculates speech rate in words per minute (WPM)."""
    if duration_s == 0:
        return 0.0  # Avoid ZeroDivisionError
    return round(len(text.split())/(duration_s/60), 2)

def estimate_pitch(audio_path):
    """Estimates the median pitch of an audio file."""
    y, sr = librosa.load(audio_path, sr=16000)
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    vals = pitches[mags > np.median(mags)] # Filter out low-magnitude pitches
    return float(np.median(vals)) if len(vals) > 0 else 0.0

def vocal_confidence(pitch_vals, rate_wpm):
    """Estimates vocal confidence based on pitch variation and speech rate."""
    var = np.var(pitch_vals) if len(pitch_vals) > 0 else 0
    rate_var = abs(rate_wpm - 130) # Deviation from an ideal WPM (e.g., 130)
    # Confidence decreases with higher pitch variance and rate deviation
    return max(0, round(1 - (var/1000 + rate_var/200), 2))

def aggregate_verbal(question, transcript, duration_s, audio_chunks_dir):
    """Aggregates all verbal metrics for a given transcript."""
    rel = score_relevance(question, transcript)
    cla = score_clarity(transcript)
    fillers = count_fillers(transcript)
    filler_ratio = sum(fillers.values()) / max(1, len(transcript.split()))
    rate = speech_rate_wpm(transcript, duration_s)

    # Estimate pitch from the first audio chunk if available
    chunk_files = [f for f in os.listdir(audio_chunks_dir) if f.endswith('.wav')]
    pitch = 0.0
    if chunk_files:
        pitch = estimate_pitch(os.path.join(audio_chunks_dir, sorted(chunk_files)[0])) # Use the first chunk

    confid = vocal_confidence([pitch], rate)
    tone = estimate_tone_from_features(rate, filler_ratio)

    return {
        'relevance': rel,
        'clarity': cla,
        'tone': tone,
        'fillers': fillers,
        'rate_wpm': rate,
        'pitch_hz': pitch,
        'vocal_confidence': confid
    }

class AudioProcessor:
    """Audio processing handler for interview recordings, consolidating all steps."""
    
    def __init__(self, audio_path, output_dir, hf_token):
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.hf_token = hf_token
        self.results = None
        
    def process(self):
        """Processes audio file through the entire pipeline: preprocessing, diarization, transcription, and metrics."""
        try:
            # 1. Preprocessing
            print(f"Starting audio preprocessing for: {self.audio_path}")
            norm_path = self.audio_path.replace('.wav', '_norm.wav')
            normalize_format(self.audio_path, norm_path)
            
            den_path = norm_path.replace('_norm.wav', '_den.wav')
            reduce_noise(norm_path, den_path)
            
            trim_path = den_path.replace('_den.wav', '_trim.wav')
            trim_silence(den_path, trim_path)
            
            # 2. Speaker Diarization
            print("Performing speaker diarization...")
            segments = diarize_audio(trim_path, self.hf_token)
            speaker_chunks = extract_segments(trim_path, segments)
            
            # 3. Transcription
            print("Transcribing speaker segments...")
            speaker_transcripts = transcribe_speaker_chunks(speaker_chunks)
            
            # 4. Get overall audio duration
            y, sr = librosa.load(trim_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # 5. Determine question and answer (simple heuristic: first speaker is question, second is answer)
            sorted_speakers = sorted(speaker_transcripts.keys())
            question_text, answer_text = "", ""
            
            if len(sorted_speakers) >= 2:
                question_text = speaker_transcripts[sorted_speakers[0]]
                answer_text = speaker_transcripts[sorted_speakers[1]]
            elif len(sorted_speakers) == 1:
                answer_text = speaker_transcripts[sorted_speakers[0]] # Assume the only speaker is the candidate

            # 6. Chunk audio for pitch analysis (if needed)
            chunks_dir = trim_path.replace('.wav', '_chunks')
            chunk_audio(trim_path, chunks_dir) # Chunks are saved to disk for aggregate_verbal
            
            # 7. Get verbal metrics for the answer
            print("Analyzing speech metrics...")
            verbal_metrics = aggregate_verbal(question_text, answer_text, duration, chunks_dir)
            
            # 8. Compile and save results
            self.results = {
                'transcripts': speaker_transcripts,
                'metrics': verbal_metrics,
                'processing_timestamp': datetime.now().isoformat()
            }
            
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

def run_audio_processing_async(session_id, username, audio_path, base_dir, hf_token):
    """
    Initiates audio processing in a background thread.
    Updates processing status files.
    """
    user_dir = os.path.join(base_dir, 'Users', username)
    session_dir = os.path.join(user_dir, 'interview', session_id)
    
    # Update processing status to 'in_progress'
    status_path = os.path.join(session_dir, 'processing_status.json')
    with open(status_path, 'w') as f:
        json.dump({
            'audio_processing': 'in_progress',
            'started_at': datetime.now().isoformat()
        }, f, indent=2)
    
    # Define the target function for the thread
    def process_thread():
        try:
            processor = AudioProcessor(audio_path, session_dir, hf_token)
            results = processor.process()
            
            # Update combined analysis file with audio analysis results
            analysis_path = os.path.join(user_dir, 'interview_analysis.json')
            if os.path.exists(analysis_path):
                with open(analysis_path, 'r') as f:
                    analysis = json.load(f)
                
                analysis['audio_analysis'] = results
                analysis['audio_processing_completed'] = True # Mark audio processing as complete
                
                with open(analysis_path, 'w') as f:
                    json.dump(analysis, f, indent=2)
            
            # Update session-specific processing status to 'completed'
            with open(status_path, 'w') as f:
                json.dump({
                    'audio_processing': 'completed',
                    'completed_at': datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error in audio processing thread: {e}")
            # Update processing status with error information
            with open(status_path, 'w') as f:
                json.dump({
                    'audio_processing': 'error',
                    'error': str(e),
                    'error_at': datetime.now().isoformat()
                }, f, indent=2)
    
    # Start the background thread
    import threading
    thread = threading.Thread(target=process_thread)
    thread.daemon = True # Allow the main program to exit even if thread is running
    thread.start()
    
    return {
        'status': 'started',
        'message': 'Audio processing started in background'
    }

