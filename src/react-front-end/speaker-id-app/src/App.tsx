import { useState } from "react";
import axios from "axios";
import { BACKEND_URL } from "./config";

function App() {
  const [name, setName] = useState("");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [result, setResult] = useState<object | null>(null);
  // const [transcript, setTranscript] = useState<string | null>(null);
  interface TranscriptEntry {
    id: string;
    name: string;
    message: string;
    audioBlob: Blob;
    speaker: string;
  }
  // const previousTranscripts:string[]  = [];
  const [previousTranscripts, setPreviousTranscripts] = useState<TranscriptEntry[]>([]);

  // Function to start recording audio  
  const startRecording = () => {
    setAudioBlob(null);
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      const mediaRecorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];

      mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        setAudioBlob(blob);
      };

      mediaRecorder.start();
      setTimeout(() => mediaRecorder.stop(), 3000); // record 3 seconds
    });
  };

  const sendToServer = async (endpoint: "enroll" | "identify") => {
    if (!audioBlob) return;

    const formData = new FormData();
    formData.append("file", audioBlob, "audio/webm");

    // Import the backend URL from config
    let url = `${BACKEND_URL}/${endpoint}`;
    if (endpoint === "enroll" && name) url += `/${name}`;

    const res = await axios.post(url, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    setResult(res.data);
    if (endpoint === "enroll") {
      transcribe(audioBlob, null).then(() => {
        console.log("Transcription after enrollment done", res.data);
      }, (err) => {
        console.error("Transcription after enrollment error", err);
      });
    }
  };

  const transcribe = async (inputAudioBlob: Blob | null, audioSampleId: string | null) => {
    if (!inputAudioBlob) return;

    const formData = new FormData();
    formData.append("file", inputAudioBlob, "audio.webm");

    try {
      const res = await axios.post(`${BACKEND_URL}/transcribe`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      // Expecting { text: "..." }
      const transcribedMessage = res.data?.text ?? JSON.stringify(res.data);
      console.log("Transcription result:", transcribedMessage);
      // setTranscript(transcribedMessage);
      if(audioSampleId == null)
        setPreviousTranscripts([{id: self.crypto.randomUUID(), name: name,  message: transcribedMessage, audioBlob: inputAudioBlob, speaker: name}, ...previousTranscripts]);
      else {
        // Update existing entry
        setPreviousTranscripts(previousTranscripts.map(elem => {
          if(elem.id === audioSampleId) {
            return {id: elem.id, name: elem.name, message: transcribedMessage, audioBlob: elem.audioBlob, speaker: elem.speaker};
          }
          return elem;
        }));
      }
    } catch (err: unknown) {
      console.error("Transcription error", err);
      let msg = "unknown error";
      if (err instanceof Error) msg = err.message;
      else if (typeof err === "string") msg = err;
      console.error("Transcription failed:", msg);
      // setTranscript("Transcription failed: " + msg);      
    }
  };

  function playThisAudio(audioBlob: Blob) {
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
  }

  const playRecording = () => {
    if (!audioBlob) {
      console.log("No audio recorded yet");
      return};
    playThisAudio(audioBlob);
  };

  return (
    <>
    <style>{`
    .app-container {
      padding: 20px;
      font-family: sans-serif;
      max-width: 900px;
      margin: auto;
    }

    h2 {
      margin-bottom: 16px;
    }

    input {
      padding: 8px;
      margin-right: 8px;
      border-radius: 4px;
      border: 1px solid #ccc;
    }

    button {
      padding: 8px 12px;
      margin: 4px;
      border-radius: 4px;
      border: none;
      cursor: pointer;
      background-color: #4f46e5;
      color: white;
    }

    button:disabled {
      background-color: #aaa;
      cursor: not-allowed;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
    }

    th, td {
      padding: 8px;
      text-align: left;
      border-bottom: 1px solid #ddd;
    }

    th {
      background-color: #5f99d3ff;
    }

    pre {
      background: #5f99d3ff;
      padding: 10px;
      border-radius: 6px;
      margin-top: 10px;
    }
  `}</style>
  
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <h2>🎤 Fun Speaker ID App</h2>

      <input
        type="text"
        placeholder="Friend's name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      
      <button onClick={() => sendToServer("enroll")} disabled={!name || !audioBlob}>Enroll</button>
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
      <br />
      <br />

      <button onClick={startRecording}>🎙️ Record 3s</button>
      <button onClick={playRecording} disabled={!audioBlob}>▶️ Play Recorded Message</button>
      <button onClick={() => sendToServer("identify")}> 👤❓Identify Me</button>
      <br />
      <br />
      
      {/* <button onClick={() => transcribe(audioBlob, null)} disabled={!audioBlob}>Transcribe</button>
      <span>{transcript}</span> */}

      <br />
      <br />

      
      
      {previousTranscripts.length > 0 && (
        <div>
          <h3>Transcription History</h3>
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Message</th>
                <th>Play</th>
                <th>Re-trascribe</th>
                <th>Delete</th>
              </tr>
            </thead>
          <tbody>
            {previousTranscripts.map((elem) => (
              <tr key={elem.id}>
                <td>({ elem.id.slice(-4)})</td>
                <td>{elem.name}</td>
                <td>{elem.message}</td>
                <td>
                  <button onClick={() => playThisAudio(elem.audioBlob)}>▶️</button>
                </td>
                <td>
                  <button onClick={() => transcribe(elem.audioBlob, elem.id)}>🔁</button>
                </td>
                <td>
                  <button
                    onClick={() => {
                      setPreviousTranscripts(previousTranscripts.filter(e => e.id !== elem.id));
                    }}
                  >
                    ❌
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
</table>

        </div>
      )}
    </div>
    </>
  );
}

export default App;
