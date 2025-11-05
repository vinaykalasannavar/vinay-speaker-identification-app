Backend STT setup

This project adds a /transcribe endpoint in `app.py` which forwards uploaded audio to Hugging Face Inference API.

Prerequisites (local transformers mode):
- Python 3.8+
- Install required packages for local inference. At minimum you'll need `transformers` and a supported `torch` build. Example:

    pip install transformers torch soundfile ffmpeg-python

  - If you have a CUDA-capable GPU and want faster inference, install a CUDA-enabled torch wheel from https://pytorch.org/.

Environment:
- AZURE_COSMOS_URI and AZURE_COSMOS_KEY are required by the project.

Running locally (Windows PowerShell example):

    $env:AZURE_COSMOS_URI = "..."
    $env:AZURE_COSMOS_KEY = "..."
    python app.py

Notes:
- The first run will download the ASR model (e.g. `openai/whisper-large-v2`) which is large (>GB). Ensure you have enough disk space.
- If you prefer using Hugging Face's hosted Inference API instead of local models, revert to the previous code and set `HF_API_TOKEN`.
- The endpoint expects a multipart/form-data POST with the audio file under the `file` field.
