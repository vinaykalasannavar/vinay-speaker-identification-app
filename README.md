# Speaker Identification App

This is a simple application designed to identify speakers from audio samples. It consists of a React-based front-end and a Python FastAPI back-end.

## Front-End

- **Framework:** React
- **Build Tool:** Vite
- **Language:** TypeScript

The front-end provides a user-friendly interface for uploading audio files and viewing identification results. Important libraries used include:

- `react` for building UI components
- `axios` for making HTTP requests to the back-end
- `vite` for fast development and build processes

## Back-End

- **Framework:** FastAPI
- **Language:** Python

The back-end handles audio processing and speaker identification. It exposes RESTful endpoints that the front-end communicates with. Key libraries include:

- `fastapi` for API development
- `uvicorn` for running the server

## Communication

The front-end communicates with the back-end via HTTP requests. When a user uploads an audio file, the front-end sends it to the FastAPI server, which processes the file and returns the identification results.

---

Feel free to explore and contribute!