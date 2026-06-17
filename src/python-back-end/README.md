# Mechanism


1. How is the training data is encoded into smaller vectors
2. How is the test data compared to the training data to "identify" a speaker  


## 1. /enroll/{name}

Steps taken:

```
┌─────────────────────┐
│ Audio Upload        │
│ (webm / opus)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ get_embedding()     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ convert_to_wav()    │
│ FFmpeg              │
│ - WAV              │
│ - Mono             │
│ - 16 kHz           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ wavfile.read()      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Torch Tensor        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Resample if needed  │
│ -> 16 kHz           │
└──────────┬──────────┘
           │
           ▼
┌────────────────────────────┐
│ SpeechBrain ECAPA-TDNN     │
│ encode_batch()             │
└──────────┬─────────────────┘
           │
           ▼
┌─────────────────────┐
│ Speaker Embedding   │
│ (~192 dimensions)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ L2 Normalize        │
│ emb /= ||emb||      │
└──────────┬──────────┘
           │
           ▼
      Normalized
       Embedding
           │
           │
           ├─────────────────────────────┐
           │                             │
           ▼                             ▼
┌──────────────────┐       ┌────────────────────┐
│ Store Embedding  │       │ extractTextFromAudio()
│ in speaker_db    │       └─────────┬──────────┘
└────────┬─────────┘                 │
         │                           ▼
         │                ┌────────────────────┐
         │                │ Whisper Tiny ASR   │
         │                └─────────┬──────────┘
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────┐
│ Store:                                   │
│ - recording id                           │
│ - embedding                              │
│ - audio_base64                           │
│ - transcript                             │
│ - timestamp                              │
└──────────────────────────────────────────┘
```

## 2. /identify

Explanation of steps:
```
┌─────────────────────┐
│ Test Audio Upload   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ get_embedding()     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Normalized Test     │
│ Embedding           │
└──────────┬──────────┘
           │
           ▼

     For each speaker
           │
           ▼

┌───────────────────────────────┐
│ Load Enrolled Embeddings      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ average_embeddings()          │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Normalize each embedding      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Stack embeddings              │
│ np.stack()                    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Compute mean vector           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Normalize average            │
│ avg /= ||avg||               │
└───────────────┬───────────────┘
                │
                ▼
         Speaker Template
                │
                ▼
┌──────────────────────────────────────┐
│ Cosine Similarity                    │
│                                      │
│ dot(test_emb, avg) /                │
│ (||test_emb|| × ||avg||)            │
└────────────────┬─────────────────────┘
                 │
                 ▼
          Similarity Score
             (-1 → +1)
                 │
                 ▼
      Keep Highest Score
                 │
                 ▼
┌─────────────────────────────┐
│ Convert to Percentage       │
│ score × 100                │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Return                      │
│                             │
│ speaker                     │
│ score (%)                   │
│ raw_similarity              │
└─────────────────────────────┘
```

Example:
```
John Denver  -> 0.92
Sarah Smith  -> 0.64
Mike Brown   -> 0.41

Winner = John Denver

92%
```

## 3. /transcribe

How does the voice to text transription work:

```
┌─────────────────────┐
│ Audio Upload        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ extractTextFromAudio()
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ convert_to_wav()    │
│ FFmpeg              │
│ Mono / 16 kHz WAV   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ wavfile.read()      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Convert to float32  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Stereo ?            │
│ Average channels    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Resample if needed  │
│ -> 16 kHz           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Whisper Tiny Model  │
│ transcribe()        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Extract Text        │
└──────────┬──────────┘
           │
           ▼
┌───────────────────────────────┐
│ Update matching recording     │
│ in speaker_db                 │
│                               │
│ message = text                │
│ timestamp = now()             │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Return                        │
│                               │
│ text                          │
│ timestamp                     │
└───────────────────────────────┘
```

## High-level architecture

This is essentially how the pipeline works:
> `speaker enrollment` → `speaker identification` → `speech transcription`  
> It is built using:
- `SpeechBrain ECAPA-TDNN` for voice recognition
- `Whisper Tiny` for speech-to-text.

```
                    ┌─────────────────┐
                    │ Audio Recording │
                    └────────┬────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │ FFmpeg Conversion  │
                  │ WAV / Mono /16 kHz │
                  └───────┬────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼

 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
 │ Enroll       │  │ Identify     │  │ Transcribe   │
 └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
        │                 │                 │
        ▼                 ▼                 ▼

 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
 │ ECAPA-TDNN   │  │ ECAPA-TDNN   │  │ Whisper Tiny │
 │ Embeddings   │  │ Embeddings   │  │ Speech->Text │
 └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
        │                 │                 │
        └─────────┬───────┴─────────────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ speaker_db       │
         │                  │
         │ embedding        │
         │ transcript       │
         │ audio            │
         │ timestamp        │
         └──────────────────┘
```