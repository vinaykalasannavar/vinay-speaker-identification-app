from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torchaudio
import io
from speechbrain.inference import SpeakerRecognition
import uvicorn
import json
import ffmpeg

app = FastAPI()

# Allow frontend (Vite dev server on port 5173)
with open("config.json", encoding="utf-8") as f:
    config = json.load(f)

print("CORS config:", config)

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

# In-memory "database"
speaker_db = {}  # {name: embedding_vector}


@app.get("/info")
async def get_info():
    return {
        "model": "ECAPA-TDNN", 
        "description": "Speaker recognition using SpeechBrain",
        "profiles": json.dumps(speaker_db)
        }


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
    
    print("Generated embedding:", emb)
    print("Embedding shape:", emb.shape)
    print("Embedding ndim:", emb.ndim)
    print("Embedding size:", emb.size)
    print("Embedding dtype:", emb.dtype)
    
    return emb

@app.post("/enroll/{name}")
async def enroll_speaker(file: UploadFile, name: str):
    audio_bytes = await file.read()
    emb = get_embedding(audio_bytes)
    speaker_db[name] = emb
    return {"message": f"Enrolled {name} successfully!"}

@app.post("/identify")
async def identify_speaker(file: UploadFile):
    if not speaker_db:
        return {"error": "No speakers enrolled yet."}

    audio_bytes = await file.read()
    test_emb = get_embedding(audio_bytes)

    best_name, best_score = None, -1
    for name, emb in speaker_db.items():
        score = np.dot(test_emb, emb.T) / (
            np.linalg.norm(test_emb) * np.linalg.norm(emb)
        )
        if score > best_score:
            best_score, best_name = score, name

    return {"speaker": best_name, "score": float(best_score)}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=config.get("host", "0.0.0.0"), 
        port=config.get("port", 8000)
    )