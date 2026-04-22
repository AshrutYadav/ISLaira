import { useState, useEffect, useRef, useCallback } from 'react';

export function useRealTimeConnection(isDetecting) {
  const [currentText, setCurrentText] = useState("");
  const [status, setStatus] = useState("🔴 No connection");
  const [history, setHistory] = useState([]);
  const [liveGesture, setLiveGesture] = useState(null); // { gesture, phrase, confidence }
  const wsRef = useRef(null);

  useEffect(() => {
    let active = true;
    let ws = null;

    const connect = () => {
      ws = new WebSocket('ws://localhost:8000/ws/stream');
      wsRef.current = ws;

      ws.onopen = () => {
        if (!active) return;
        setStatus(isDetecting ? "🟢 Detecting" : "🟡 Connected (Paused)");
      };

      ws.onmessage = (event) => {
        if (!active) return;
        try {
          const data = JSON.parse(event.data);

          if (data.type === "gesture") {
            // Live gesture detection — update in real time
            setStatus(data.status || "🟢 Detecting");
            setLiveGesture({ gesture: data.gesture, phrase: data.phrase, confidence: data.confidence });

          } else if (data.type === "translation") {
            // Confirmed gesture — add to history
            setCurrentText(data.text);
            setLiveGesture(null);
            setHistory(prev => {
              if (prev.length > 0 && prev[0].text === data.text) return prev;
              return [{ id: Date.now(), text: data.text, gesture: data.gesture }, ...prev];
            });

          } else if (data.type === "status") {
            setStatus(data.status);
            if (data.status.includes("No hands")) {
              setLiveGesture(null);
            }
          }
        } catch (e) {
          console.error("WS parse error", e);
        }
      };

      ws.onclose = () => {
        if (!active) return;
        setStatus("🔴 Disconnected (Retrying...)");
        setTimeout(() => { if (active) connect(); }, 2000);
      };

      ws.onerror = () => {
        setStatus("🔴 Connection error");
      };
    };

    connect();

    return () => {
      active = false;
      if (ws) { ws.onclose = null; ws.close(); }
    };
  }, []);

  useEffect(() => {
    if (!isDetecting && wsRef.current?.readyState === WebSocket.OPEN) {
      setStatus("🟡 Connected (Paused)");
      setLiveGesture(null);
    } else if (isDetecting && wsRef.current?.readyState === WebSocket.OPEN) {
      setStatus("🟢 Detecting");
    }
  }, [isDetecting]);

  const clearSession = useCallback(() => {
    setCurrentText("");
    setHistory([]);
    setLiveGesture(null);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "clear" }));
    }
  }, []);

  const sendFrame = useCallback((base64Image) => {
    if (isDetecting && wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ frame: base64Image }));
    }
  }, [isDetecting]);

  return { currentText, setCurrentText, status, history, liveGesture, clearSession, sendFrame };
}
