import { useState, useEffect } from "react";
import axios from "axios";
import { BACKEND_URL } from "./config";

interface TranscriptEntry {
  id: string;
  name: string;
  message: string;
  audioBlob: Blob | null;
  speaker: string;
}

function App() {
  const [name, setName] = useState("");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [result, setResult] = useState<object | null>(null);
  const [previousTranscripts, setPreviousTranscripts] = useState<TranscriptEntry[]>([]);

  useEffect(() => {
    loadTranscripts();
  }, []);

  const loadTranscripts = async () => {
    try {
      const res = await axios.get(`${BACKEND_URL}/transcripts`);

      const loaded = res.data.map((item: any) => {
        let blob: Blob | null = null;

        if (item.audio_base64) {
          const byteChars = atob(item.audio_base64);
          const byteNumbers = new Array(byteChars.length);

          for (let i = 0; i < byteChars.length; i++) {
            byteNumbers[i] = byteChars.charCodeAt(i);
          }

          const byteArray = new Uint8Array(byteNumbers);
          blob = new Blob([byteArray], { type: "audio/webm" });
        }

        return {
          id: item.id,
          name: item.name,
          message: item.message,
          audioBlob: blob,
          speaker: item.name,
        };
      });

      setPreviousTranscripts(loaded);
    } catch (err) {
      console.error("Failed to load transcripts", err);
    }
  };

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
      setTimeout(() => mediaRecorder.stop(), 3000);
    });
  };

 const sendToServer = async (endpoint: "enroll" | "identify") => {
   if (!audioBlob) return;

   const formData = new FormData();
   formData.append("file", audioBlob, "audio.webm");

   let recordingId: string | null = null;

   if (endpoint === "enroll") {
     recordingId = crypto.randomUUID();
     formData.append("recording_id", recordingId);
   }

   let url = `${BACKEND_URL}/${endpoint}`;
   if (endpoint === "enroll" && name) url += `/${name}`;

   const res = await axios.post(url, formData);
   setResult(res.data);

   // ✅ ADD DIRECTLY TO HISTORY (no extra API call)
   if (endpoint === "enroll") {
     const text = res.data.text ?? "";

     setPreviousTranscripts((prev) => [
       {
         id: recordingId!,
         name: name,
         message: text,
         audioBlob: audioBlob,
         speaker: name,
       },
       ...prev,
     ]);
   }
 };

  const transcribe = async (
    inputAudioBlob: Blob | null,
    audioSampleId: string | null,
    speakerName: string = name,
    recordingId: string | null = null
  ) => {
    if (!inputAudioBlob) return;

    const formData = new FormData();
    formData.append("file", inputAudioBlob, "audio.webm");

    const id = recordingId || crypto.randomUUID();
    formData.append("recording_id", id);
    formData.append("speaker_name", speakerName);

    const res = await axios.post(`${BACKEND_URL}/transcribe`, formData);
    const text = res.data.text ?? "";

    if (audioSampleId == null) {
      setPreviousTranscripts((prev) => [
        {
          id,
          name: speakerName,
          message: text,
          audioBlob: inputAudioBlob,
          speaker: speakerName,
        },
        ...prev,
      ]);
    } else {
      setPreviousTranscripts((prev) =>
        prev.map((entry) =>
          entry.id === audioSampleId ? { ...entry, message: text } : entry
        )
      );
    }
  };

  const deleteRecording = async (id: string) => {
    try {
      await axios.delete(`${BACKEND_URL}/recording/${id}`);
      setPreviousTranscripts((prev) => prev.filter((x) => x.id !== id));
    } catch (err) {
      console.error("Delete failed", err);
    }
  };

  const clearAll = async () => {
    const confirmClear = window.confirm("Are you sure you want to delete ALL recordings?");
    if (!confirmClear) return;

    try {
      await axios.delete(`${BACKEND_URL}/transcripts?confirm=true`);
      setPreviousTranscripts([]);
      setResult(null);
    } catch (err) {
      console.error("Clear all failed", err);
    }
  };

  const playThisAudio = (blob: Blob) => {
    const url = URL.createObjectURL(blob);
    new Audio(url).play();
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
          background-color: #2ba276;
          color: white;
        }

        button:hover {
          opacity: 0.9;
        }

        button:disabled {
          background-color: #aaa;
          cursor: not-allowed;
        }

        .danger {
          background-color: #d9534f;
        }

        table {
          width: 100%;
          border-collapse: collapse;
          margin-top: 16px;
        }

        th, td {
          padding: 8px;
          text-align: left;
          border: 1px solid #847f7f;
        }

        th {
          background-color: #5f99d3;
        }

        pre {
          background: #5f89d3;
          padding: 10px;
          color: #f8f1ea;
          border-radius: 6px;
          margin-top: 10px;
        }
      `}</style>

      <div className="app-container">
        <h2>🎤 Fun Speaker ID App</h2>

        <input
          type="text"
          placeholder="Friend's name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />

        <button onClick={() => sendToServer("enroll")} disabled={!name || !audioBlob}>
          Enroll
        </button>

        <button onClick={startRecording}>🎙️ Record 3s</button>

        <button onClick={() => audioBlob && playThisAudio(audioBlob)} disabled={!audioBlob}>
          ▶️ Play Recorded Message
        </button>

        <button onClick={() => sendToServer("identify")}>
          👤 Identify Me
        </button>

        <button className="danger" onClick={clearAll} disabled={!(previousTranscripts.length > 0)}>
          🧹 Clear All
        </button>

        {result && <pre>{JSON.stringify(result, null, 2)}</pre>}

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
                  <th>Re-transcribe</th>
                  <th>Delete</th>
                </tr>
              </thead>

              <tbody>
                {previousTranscripts.map((entry) => (
                  <tr key={entry.id}>
                    <td>({entry.id.slice(-4)})</td>
                    <td>{entry.name}</td>
                    <td>{entry.message}</td>

                    <td>
                      <button onClick={() => entry.audioBlob && playThisAudio(entry.audioBlob)}>
                        ▶️
                      </button>
                    </td>

                    <td>
                      <button onClick={() => transcribe(entry.audioBlob, entry.id, entry.name)}>
                        🔁
                      </button>
                    </td>

                    <td>
                      <button className="danger" onClick={() => deleteRecording(entry.id)}>
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