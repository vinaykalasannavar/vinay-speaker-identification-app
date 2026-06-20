# Speaker Identification App

A simple app designed to identify speakers from audio samples.

* The front-end is a React-based UI
* The back-end is built as a Python FastAPI.

## Front-End
### Tech spec:
- **Framework:** React
- **Build Tool:** Vite
- **Language:** TypeScript

The front-end provides a user-friendly interface for recording audio samples (training data).
You can identify a user later (test data).

### Key libraries used:
- `react` for building UI components
- `axios` for making HTTP requests to the back-end
- `vite` for fast development and build processes

## Back-End

The back-end handles audio processing and speaker identification. It exposes RESTful endpoints that the front-end communicates with.

### Tech spec
- **Framework:** FastAPI
- **Language:** Python

### Key libraries used:
- `fastapi` for API development
- `uvicorn` for running the server

## Communication

The front-end communicates with the back-end via HTTP requests.
User records audio samples, the front-end sends it to the FastAPI server.
The Python based API embeds the audio samples and stores them (currently stores it in-memory - later to be persisted in a databse).
User can record another sample - to identify the result against the stored users samples.

---

Feel free to explore and use the code if it helps you!