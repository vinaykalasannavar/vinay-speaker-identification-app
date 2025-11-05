import { useState } from "react";
import axios from "axios";
import { BACKEND_URL } from "./config";

function App() {
  const [name, setName] = useState("");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [result, setResult] = useState<object | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  interface TranscriptEntry {
    id: string;
    message: string;
    audioBlob: Blob;
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
      transcribe().then(() => {
        console.log("Transcription after enrollment done");
      }, (err) => {
        console.error("Transcription after enrollment error", err);
      });
    }
  };

  const transcribe = async () => {
    if (!audioBlob) return;

    const formData = new FormData();
    formData.append("file", audioBlob, "audio.webm");

    try {
      const res = await axios.post(`${BACKEND_URL}/transcribe`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      // Expecting { text: "..." }
      const transcribedMessage = res.data?.text ?? JSON.stringify(res.data);
      setTranscript(transcribedMessage);

      setPreviousTranscripts([{id: self.crypto.randomUUID(),  message: transcribedMessage, audioBlob: audioBlob}, ...previousTranscripts]);
    } catch (err: unknown) {
      console.error("Transcription error", err);
      let msg = "unknown error";
      if (err instanceof Error) msg = err.message;
      else if (typeof err === "string") msg = err;
      setTranscript("Transcription failed: " + msg);      
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
      
      <button onClick={transcribe} disabled={!audioBlob}>Transcribe</button>
      <span>{transcript}</span>

      <br />
      <br />

      
      
      {previousTranscripts.length > 0 && (
        <div>
          <h3>Transcription History</h3>
          {/* <ul>
            {previousTranscripts.map((elem) => (
              <li key="{elem.id}">
                <span>{elem.id}</span>
                <span>{elem.message}</span>
                <button onClick={()=>{playThisAudio(elem.audioBlob)}}>▶️</button>
                <button onClick={()=>{
                  setPreviousTranscripts(previousTranscripts.filter(e=>e.id!==elem.id));
                }}>❌</button>
              </li>
            ))}
          </ul> */}

          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Message</th>
                <th>Play</th>
                <th>Delete</th>
              </tr>
            </thead>
          <tbody>
            {previousTranscripts.map((elem) => (
              <tr key={elem.id}>
                <td>{elem.id}</td>
                <td>{elem.message}</td>
                <td>
                  <button onClick={() => playThisAudio(elem.audioBlob)}>▶️</button>
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
  );
}

export default App;
