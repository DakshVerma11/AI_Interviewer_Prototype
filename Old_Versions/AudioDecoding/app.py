import os
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"
from flask import Flask, render_template, request
from dotenv import load_dotenv
import time
import librosa

from analysis_audio import (
    extract_audio_ffmpeg, normalize_format, reduce_noise,
    trim_silence, chunk_audio, transcribe_with_retries,
    aggregate_verbal
)
from diarize import diarize_audio, extract_segments, transcribe_speaker_chunks

load_dotenv()
app = Flask(__name__)
UPLOAD_FOLDER = 'C:\\Users\\Dinesh\\AI-Interviewer\\uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_audio', methods=['POST'])
def submit_audio():
    audio = request.files['audio']
    ts = time.time()
    audio_path = os.path.join(UPLOAD_FOLDER, f"{ts}_{audio.filename}")
    audio.save(audio_path)

    norm = audio_path.replace('.wav', '_norm.wav')
    normalize_format(audio_path, norm)
    den = norm.replace('_norm.wav','_den.wav')
    reduce_noise(norm, den)
    trim = den.replace('_den.wav','_trim.wav')
    trim_silence(den, trim)

    hf_token = os.getenv("HF_TOKEN")
    segments = diarize_audio(trim, hf_token)
    speaker_chunks = extract_segments(trim, segments)
    speaker_transcripts = transcribe_speaker_chunks(speaker_chunks)

    print("Speaker chunks:", list(speaker_transcripts.keys()))
    print("Transcripts:", speaker_transcripts)

    sorted_speakers = sorted(speaker_transcripts.keys())
    question, answer = "", ""

    if len(sorted_speakers) >= 2:
        question = speaker_transcripts[sorted_speakers[0]]
        answer = speaker_transcripts[sorted_speakers[1]]
    elif len(sorted_speakers) == 1:
        answer = speaker_transcripts[sorted_speakers[0]]

    chunks_dir = trim.replace('.wav','_chunks')
    chunks = chunk_audio(trim, chunks_dir)
    y, sr = librosa.load(trim, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    verbal = aggregate_verbal(question, answer, duration, chunks_dir)
    return render_template('result.html', result=verbal, transcripts=speaker_transcripts)
    #return render_template('result.html', result=verbal, transcript=answer)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
