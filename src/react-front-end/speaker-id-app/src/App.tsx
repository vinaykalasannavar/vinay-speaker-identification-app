import { useState } from "react";
import axios from "axios";
import { BACKEND_URL } from "./config";

function App() {
  const [name, setName] = useState("");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [result, setResult] = useState<object | null>(null);

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
    console.log(`res = `, res);
    setResult(res.data);
  };

  const playRecording = () => {
    if (!audioBlob) {
      console.log("No audio recorded yet");
      return};
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
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
      <br />
      <br />

      <button onClick={startRecording}>🎙️ Record 3s</button>
      <button onClick={playRecording} disabled={!audioBlob}>▶️ Play Recorded Message</button>

      <br />
      <br />

      <button onClick={() => sendToServer("enroll")}>Enroll</button>
      <button onClick={() => sendToServer("identify")}>Identify</button>
      <br />
      <br />

      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}

export default App;
