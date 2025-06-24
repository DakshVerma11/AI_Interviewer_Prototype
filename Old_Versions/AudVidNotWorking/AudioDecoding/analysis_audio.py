import os, subprocess, time, logging
import whisper
import librosa, soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, effects
import numpy as np

if not hasattr(np,'float'):
    np.float = float

# --- Extraction ---
def extract_audio_ffmpeg(video_path, audio_path):
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-ac', '1', '-ar', '16000', '-sample_fmt', 's16', audio_path
    ]
    subprocess.run(cmd, check=True)

# --- Preprocessing ---
def normalize_format(input_path, output_path):
    y, sr = librosa.load(input_path, sr=16000)
    sf.write(output_path, y, 16000, subtype='PCM_16')

def reduce_noise(wav_path, denoised_path):
    y, sr = librosa.load(wav_path, sr=16000)
    noisy_part = y[0:int(sr*0.5)]
    reduced = nr.reduce_noise(y=y, y_noise=noisy_part, sr=sr)
    sf.write(denoised_path, reduced, sr)

def trim_silence(input_path, trimmed_path):
    audio = AudioSegment.from_wav(input_path)
    trimmed = effects.strip_silence(audio, silence_thresh=-40)
    trimmed.export(trimmed_path, format='wav')

# --- Chunking ---
def chunk_audio(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    audio = AudioSegment.from_wav(path)
    dur_ms = len(audio)
    step_ms = 25000
    chunks = []
    for start in range(0, dur_ms, step_ms):
        end = min(start + 30000, dur_ms)
        chunk = audio[start:end]
        fn = os.path.join(out_dir, f"chunk_{start}.wav")
        chunk.export(fn, format='wav')
        chunks.append(fn)
        if end == dur_ms: break
    return chunks

# --- Transcription ---
def transcribe_with_retries(chunk_paths):
    model = whisper.load_model('base')
    transcripts, confs = [], []
    for path in chunk_paths:
        for attempt in range(3):
            try:
                res = model.transcribe(path, fp16=False)
                transcripts.append(res['text'].strip())
                confs.append(res['segments'][-1]['avg_logprob'])
                break
            except Exception as e:
                logging.warning(f"Whisper error on {path}: {e}")
                time.sleep(1)
        else:
            transcripts.append('')
            confs.append(0)
    full = ' '.join(transcripts)
    avg_conf = float(np.mean([c for c in confs if c is not None]))
    return full, avg_conf

# --- NLP Scoring ---
FILLERS = ['uh', 'um', 'like', 'you know', 'so']

def count_fillers(text):
    words = text.lower().split()
    return {f: words.count(f) for f in FILLERS}

def score_relevance(question, answer):
    # Naive lexical overlap score (as fallback for no OpenAI)
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    if not q_words: return 0
    return round(len(q_words & a_words) / len(q_words), 2)

def score_clarity(text):
    words = text.split()
    filler_count = sum(text.lower().count(f) for f in FILLERS)
    return round(1 - min(1.0, filler_count / max(1, len(words))), 2)

def estimate_tone_from_features(rate_wpm, filler_ratio):
    if rate_wpm > 160 and filler_ratio < 0.05:
        label = "Confident"
    elif filler_ratio > 0.15:
        label = "Hesitant"
    else:
        label = "Neutral"
    confidence = round(min(1.0, max(0.0, 1 - filler_ratio * 2)), 2)
    return {"label": label, "confidence": confidence}

def speech_rate_wpm(text, duration_s):
    if duration_s == 0:
        return 0.0  # Avoid ZeroDivisionError
    return round(len(text.split())/(duration_s/60), 2)

def estimate_pitch(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    vals = pitches[mags > np.median(mags)]
    return float(np.median(vals)) if len(vals) > 0 else 0.0

def vocal_confidence(pitch_vals, rate_wpm):
    var = np.var(pitch_vals)
    rate_var = abs(rate_wpm - 130)
    return max(0, round(1 - (var/1000 + rate_var/200), 2))

def aggregate_verbal(question, transcript, duration_s, audio_chunks_dir):
    rel = score_relevance(question, transcript)
    cla = score_clarity(transcript)
    fillers = count_fillers(transcript)
    filler_ratio = sum(fillers.values()) / max(1, len(transcript.split()))
    rate = speech_rate_wpm(transcript, duration_s)

    chunk_files = os.listdir(audio_chunks_dir)
    if not chunk_files:
        pitch = 0.0
    else:
        pitch = estimate_pitch(os.path.join(audio_chunks_dir, chunk_files[0]))

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