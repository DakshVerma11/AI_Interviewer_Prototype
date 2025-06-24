import os
import tempfile
import whisper
from pydub import AudioSegment
from pyannote.audio import Pipeline

def diarize_audio(audio_path, hf_token):
    if not hf_token:
        raise ValueError("Missing Hugging Face token for diarization.")

    print(f"[DEBUG] Starting diarization on: {audio_path}")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=hf_token
        )
        diarization = pipeline(audio_path)
    except Exception as e:
        raise RuntimeError(f"Diarization failed: {e}")

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"[SEGMENT] Speaker: {speaker} | Start: {turn.start:.2f}s | End: {turn.end:.2f}s")
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })

    return segments

def extract_segments(audio_path, segments):
    audio = AudioSegment.from_wav(audio_path)
    speaker_audio = {}
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        speaker = seg["speaker"]
        chunk = audio[start_ms:end_ms]
        speaker_audio[speaker] = speaker_audio.get(speaker, AudioSegment.empty()) + chunk
    return speaker_audio

def transcribe_speaker_chunks(speaker_audio, model_size="base"):
    model = whisper.load_model(model_size)
    results = {}

    for speaker, audio in speaker_audio.items():
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            audio.export(temp_path, format="wav")

        print(f"[TRANSCRIBE] Transcribing audio for {speaker}...")
        try:
            result = model.transcribe(temp_path, fp16=False)
            results[speaker] = result["text"].strip()
        except Exception as e:
            print(f"[ERROR] Whisper failed on {speaker}: {e}")
            results[speaker] = ""
        finally:
            os.remove(temp_path)

    return results
