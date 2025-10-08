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
speaker_db = {}  # {name: [embedding_vector]}



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


def get_speaker(name):
    # TODO: Optimize with a direct read from Cosmos DB
    # query = f"SELECT * FROM c WHERE c.name = '{name}'"
    # items = list(container.query_items(query=query, enable_cross_partition_query=True))
    items = [speaker_db[name]] if name in speaker_db else []
    return items[0] if items else None

def save_db():
    """Persist speaker_db to disk."""
    # TODO: Implement persistent storage, to Azure Cosmos DB
    # with open(DATA_FILE, "wb") as f:
    #     pickle.dump(speaker_db, f)
    return False

def save_speaker(name, embeddings):
    # print(f'VINAY_DEBUG: Saving speaker {name}')
    # item = {
    #     "id": name,  # unique ID
    #     "name": name,
    #     "embeddings": embeddings  # list of embedding arrays (lists)
    # }
    # print(f'VINAY_DEBUG: Item to save: {item}')
    
    # Update in-memory DB for quick access
    if name not in speaker_db:
        speaker_db[name] = []
    
    # print(f'VINAY_DEBUG: Current embeddings for {name}: {speaker_db[name]}')
    speaker_db[name].append(embeddings.tolist())  # store as list for JSON/pickle safety
    # print(f'VINAY_DEBUG: Updated embeddings for {name}: {speaker_db[name]}')
    
    save_db()
    
    # TODO: Save to Cosmos DB
    # item_json = json.dumps(item, cls=json.JSONEncoder)
    # container.upsert_item(item_json)
    # container.upsert_item(item)  # Save to Cosmos DB

# -------------------------------
# API ROUTES
# -------------------------------

@app.get("/info")
async def get_info():
    return {
        "model": "ECAPA-TDNN", 
        "description": "Speaker recognition using SpeechBrain",
        "profiles": json.dumps(speaker_db)
        }

@app.post("/enroll/{name}")
async def enroll_speaker(file: UploadFile, name: str):
    audio_bytes = await file.read()
    emb = get_embedding(audio_bytes)
    
    speaker = get_speaker(name)
    
    if speaker:
        # print(f'VINAY_DEBUG: found existing speaker {name}. Appending new embedding.')
        # speaker["embeddings"].append(emb)
        save_speaker(name, emb)
        # print(f'VINAY_DEBUG: speaker_db now has {speaker_db[name]} samples for {name}.')
        # count = len(speaker[name])
        count = 'MORE_THAN_ONE' # Placeholder until we implement persistent storage
    else:
        # print(f'VINAY_DEBUG: No existing speaker found. Creating new entry for {name}.')
        save_speaker(name, emb)
        count = 1

    return {"message": f"Added sample {count} for {name}"}

@app.post("/identify")
async def identify_speaker(file: UploadFile):
    if not speaker_db:
        return {"error": "No speakers enrolled yet."}

    audio_bytes = await file.read()
    test_emb = get_embedding(audio_bytes)

    # TODO: Optimize with a direct read from Cosmos DB
    # query = "SELECT * FROM c"
    # speakers = list(container.query_items(query=query, enable_cross_partition_query=True))
    
    # For now, use in-memory DB
    speakers = speaker_db.values()

    best_name, best_score = None, -1    
    # for sp in speakers:
    #     print(f'VINAY_DEBUG: Contents of sp={sp}')
    #     embeddings = [np.array(e) for e in sp["embeddings"]]
    #     avg_emb = average_embeddings(embeddings)
    #     score = float(np.dot(test_emb, avg_emb))
    #     if score > best_score:
    #         best_score, best_name = score, sp["name"]

    for name, embeddings_list in speaker_db.items():
        # print(f'VINAY_DEBUG: Evaluating speaker {name} with {(embeddings_list)} embeddings.')

        embeddings = [np.array(e) for e in embeddings_list]
        avg_emb = average_embeddings(embeddings)
        score = float(np.dot(test_emb, avg_emb.T))
        if score > best_score:
            best_score, best_name = score, name

    return {"speaker": best_name, "score": best_score}

@app.get("/speakers")
def list_speakers():
    # TODO: Optimize with a direct read from Cosmos DB
    # query = "SELECT c.name, ARRAY_LENGTH(c.embeddings) AS count FROM c"
    # items = list(container.query_items(query=query, enable_cross_partition_query=True))    
    # items = [{"name": name, "count": len(data["embeddings"])} for name, data in speaker_db.items()]
    
    # print(f'VINAY_DEBUG: Current speaker_db contents: {speaker_db.items()}')
    # items = [{"name": name, "value": data} for name, data in speaker_db.items()]
    
    # print(f'VINAY_DEBUG: list_speakers returning {items}')

    return {item: data for item, data in speaker_db.items()}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=config.get("host", "0.0.0.0"), 
        port=config.get("port", 8000)
    )