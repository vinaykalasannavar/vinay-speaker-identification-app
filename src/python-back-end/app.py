from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torchaudio
import io
import ffmpeg
import base64
from speechbrain.inference import SpeakerRecognition

import uvicorn
import json
import ffmpeg
from azure.cosmos import CosmosClient, PartitionKey
import os

# -------------------------------
# CONFIG
# -------------------------------

# Allow frontend (Vite dev server on port 5173)
with open("config.json", encoding="utf-8") as f:
    config = json.load(f)

AZURE_COSMOS_URI = os.environ['AZURE_COSMOS_URI']
AZURE_COSMOS_KEY = os.environ['AZURE_COSMOS_KEY']

# Local ASR using Hugging Face transformers (Whisper)
try:
    from transformers import pipeline
    import torch as _torch

    _device = 0 if _torch.cuda.is_available() else -1
    # NOTE: this will download the model the first time it's run and can be large.
    # Use environment override if provided; default to the latest Whisper model
    ASR_MODEL_NAME = os.environ.get("ASR_MODEL_NAME", "openai/whisper-tiny")
    try:
        asr_pipeline = pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME, device=_device, generate_kwargs={"language": "english", "task": "transcribe"})
    except Exception as _e:
        # If model fails to load, set pipeline to None and surface error at request time.
        asr_pipeline = None
        print(f"Warning: failed to initialize ASR pipeline: {_e}")
except Exception as _e:
    asr_pipeline = None
    print(f"Warning: transformers pipeline not available: {_e}")

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
    wav_data, _ = process.communicate(input=in_buffer.read())
    out_buffer.write(wav_data)
    out_buffer.seek(0)
    return out_buffer

def get_embedding(audio_bytes):
    wav_buffer = convert_to_wav(audio_bytes)

    waveform, sr = torchaudio.load(wav_buffer)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
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

    speaker_db.setdefault(name, []).append({
        "id": recording_id,
        "embedding": emb.tolist(),
        "audio_base64": audio_base64,
        "message": text
    })

    count = len(speaker_db[name])

    return {
        "message": f"Added sample {count} for {name}",
        "recording_id": recording_id,
        "text": text
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

    if speaker_name in speaker_db:
        for r in speaker_db[speaker_name]:
            if r["id"] == recording_id:
                r["message"] = text

    return {"text": text}

def extractTextFromAudio(audio_bytes):
    wav = convert_to_wav(audio_bytes)
    wav.seek(0)

    result = asr_pipeline(wav)
    text = result["text"]
    return text


@app.get("/transcripts")
def transcripts():
    out = []
    for name, recs in speaker_db.items():
        for r in recs:
            out.append({
                "id": r["id"],
                "name": name,
                "message": r["message"],
                "audio_base64": r["audio_base64"]
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