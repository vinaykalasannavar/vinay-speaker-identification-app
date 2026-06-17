from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torchaudio
import io
import ffmpeg
import base64
from speechbrain.inference import SpeakerRecognition
from scipy.io import wavfile

import uvicorn
import json
import ffmpeg
from azure.cosmos import CosmosClient, PartitionKey
import os
import datetime

# -------------------------------
# CONFIG
# -------------------------------

# Allow frontend (Vite dev server on port 5173)
with open("config.json", encoding="utf-8") as f:
    config = json.load(f)

AZURE_COSMOS_URI = os.environ['AZURE_COSMOS_URI']
AZURE_COSMOS_KEY = os.environ['AZURE_COSMOS_KEY']

# Local ASR using OpenAI Whisper
whisper_asr_model = None
try:
    import whisper
    whisper_asr_model = whisper.load_model("tiny")
    print("ASR model loaded successfully")
except Exception as _e:
    whisper_asr_model = None
    print(f"Warning: failed to initialize ASR model: {_e}")

DATABASE_NAME = config.get("database_name", "speakerdb")
CONTAINER_NAME = config.get("container_name", "speakers")

# -------------------------------
# APP INIT
# -------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config["allow_origins"],
    allow_methods=config["allow_methods"],
    allow_headers=config["allow_headers"],
)

# Load pretrained ECAPA-TDNN model
spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# -------------------------------
# COSMOS SETUP
# -------------------------------
client = CosmosClient(AZURE_COSMOS_URI, credential=AZURE_COSMOS_KEY)

db = client.create_database_if_not_exists(id=DATABASE_NAME)
container = db.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/name")
)

# In-memory "database"
speaker_db = {}


# -------------------------------
# UTILITIES
# -------------------------------
def convert_to_wav(audio_bytes):
    """Convert webm/opus bytes to wav bytes using ffmpeg."""
    in_buffer = io.BytesIO(audio_bytes)
    out_buffer = io.BytesIO()
    process = (
        ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar='16k')
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
    )
    wav_data, stderr = process.communicate(input=in_buffer.read())
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed converting audio: {stderr.decode('utf-8', errors='ignore')}")
    out_buffer.write(wav_data)
    out_buffer.seek(0)
    return out_buffer


def get_embedding(audio_bytes):
    wav_buffer = convert_to_wav(audio_bytes)
    wav_buffer.seek(0)
    
    # Load wav using scipy instead of torchaudio to avoid torchcodec issues
    sample_rate, waveform_numpy = wavfile.read(wav_buffer)
    
    # Convert numpy array to torch tensor
    waveform = torch.from_numpy(waveform_numpy.copy()).float()
    
    # Handle mono vs stereo
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] > 1:
        # Take first channel if stereo
        waveform = waveform[0].unsqueeze(0)
    
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    emb = spkrec.encode_batch(waveform)
    emb = emb.mean(dim=0).detach().cpu().numpy()
    
    # print("Generated embedding:")
    # print("Embedding shape:", emb.shape)
    # print("Embedding ndim:", emb.ndim)
    # print("Embedding size:", emb.size)
    # print("Embedding dtype:", emb.dtype)
    
    return emb


def average_embeddings(embeddings):
    
    # print(f'VINAY_DEBUG: Averaging embeddings shape', embeddings.shape if isinstance(embeddings, np.ndarray) else 'list of length ' + str(len(embeddings)))
    arr = np.stack(embeddings)
    # print(f'VINAY_DEBUG: Stacked embeddings shape', arr.shape)

    avg = arr.mean(axis=0)
    print(f'VINAY_DEBUG: Mean embedding shape', avg[0:5])
    avg /= np.linalg.norm(avg)
    print(f'VINAY_DEBUG: Averaged embedding shape', avg[0:5])
    return avg


@app.post("/enroll/{name}")
async def enroll(file: UploadFile, name: str, recording_id: str = Form(...)):
    audio_bytes = await file.read()
    emb = get_embedding(audio_bytes)

    audio_base64 = base64.b64encode(audio_bytes).decode()
    text = extractTextFromAudio(audio_bytes)
    timestamp = datetime.datetime.now().isoformat()

    speaker_db.setdefault(name, []).append({
        "id": recording_id,
        "embedding": emb.tolist(),
        "audio_base64": audio_base64,
        "message": text,
        "timestamp": timestamp
    })

    count = len(speaker_db[name])

    return {
        "message": f"Added sample {count} for {name}",
        "recording_id": recording_id,
        "text": text,
        "timestamp": timestamp
    }


@app.post("/identify")
async def identify(file: UploadFile):
    if not speaker_db:
        return {"error": "No speakers enrolled"}

    audio_bytes = await file.read()
    test_emb = get_embedding(audio_bytes)

    best_name, best_score = None, -1

    for name, recs in speaker_db.items():
        embeddings = [np.array(r["embedding"]) for r in recs]
        avg = average_embeddings(embeddings)
        score = float(np.dot(test_emb, avg.T))

        if score > best_score:
            best_score, best_name = score, name

    return {"speaker": best_name, "score": best_score}


@app.post("/transcribe")
async def transcribe(file: UploadFile, recording_id: str = Form(...), speaker_name: str = Form("unknown")):
    audio_bytes = await file.read()
    text = extractTextFromAudio(audio_bytes)
    timestamp = datetime.datetime.now().isoformat()

    if speaker_name in speaker_db:
        for r in speaker_db[speaker_name]:
            if r["id"] == recording_id:
                r["message"] = text
                r["timestamp"] = timestamp

    return {"text": text, "timestamp": timestamp}

def extractTextFromAudio(audio_bytes):
    if whisper_asr_model is None:
        print("ASR model not available")
        return "[ASR unavailable]"

    wav_file = convert_to_wav(audio_bytes)
    wav_file.seek(0)

    try:
        sample_rate, audio_numpy_array = wavfile.read(wav_file)
        if audio_numpy_array.dtype.kind in 'iu':
            audio = audio_numpy_array.astype(np.float32) / np.iinfo(audio_numpy_array.dtype).max
        else:
            audio = audio_numpy_array.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != 16000:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            audio_resampled = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)
            audio = audio_resampled.squeeze(0).numpy()

        result = whisper_asr_model.transcribe(audio, language="en")
        text = result.get("text", "[No transcription]").strip()
        return text if text else "[No speech detected]"
    except Exception as e:
        print(f"ASR transcription error: {e}")
        return f"[Transcription error]"

@app.get("/transcripts")
def transcripts():
    out = []
    for name, recs in speaker_db.items():
        for r in recs:
            out.append({
                "id": r["id"],
                "name": name,
                "message": r["message"],
                "audio_base64": r["audio_base64"],
                "timestamp": r.get("timestamp", "")
            })
    return out


# ✅ DELETE SINGLE RECORDING
@app.delete("/recording/{recording_id}")
def delete_recording(recording_id: str):
    found = False

    for name in list(speaker_db.keys()):
        speaker_db[name] = [r for r in speaker_db[name] if r["id"] != recording_id]

        if not speaker_db[name]:
            del speaker_db[name]

        else:
            found = True

    if not found:
        raise HTTPException(status_code=404, detail="Recording not found")

    return {"message": "Deleted"}


# ✅ CLEAR ALL
@app.delete("/transcripts")
def clear_all(confirm: bool = False):
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")

    speaker_db.clear()
    return {"message": "All data cleared"}

    
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=config.get("host", "0.0.0.0"), 
        port=config.get("port", 8000)
    )