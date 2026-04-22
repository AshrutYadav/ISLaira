import { useState, useEffect, useRef } from 'react';

const DUMMY_PHRASES = [
  "Hello, how are you today?",
  "I am doing well, thank you.",
  "Can you help me with this?",
  "What is the weather like usually?",
  "It is nice to meet you.",
  "I need to go to the hospital.",
  "Where is the nearest pharmacy?",
  "Thank you for your assistance.",
  "Sign language is beautiful."
];

export function useRealTimeSimulation(isDetecting) {
  const [currentText, setCurrentText] = useState("");
  const [status, setStatus] = useState("🔴 No hands detected"); // 🔴, 🟡, 🟢
  const [history, setHistory] = useState([]);
  
  const simulationIntervalRef = useRef(null);
  const wordIntervalRef = useRef(null);

  useEffect(() => {
    if (!isDetecting) {
      setStatus("🔴 Stopped");
      clearInterval(simulationIntervalRef.current);
      clearInterval(wordIntervalRef.current);
      return;
    }

    setStatus("🟢 Detecting");

    simulationIntervalRef.current = setInterval(() => {
      // Simulate processing a sequence
      setStatus("🟡 Processing");
      
      setTimeout(() => {
        // Pick a random phrase and "stream" it word by word
        const phrase = DUMMY_PHRASES[Math.floor(Math.random() * DUMMY_PHRASES.length)];
        const words = phrase.split(" ");
        let wordIndex = 0;
        
        setCurrentText("");
        setStatus("🟢 Detecting");

        clearInterval(wordIntervalRef.current);
        wordIntervalRef.current = setInterval(() => {
          if (wordIndex < words.length) {
            setCurrentText(prev => prev + (prev.length > 0 ? " " : "") + words[wordIndex]);
            wordIndex++;
          } else {
            clearInterval(wordIntervalRef.current);
            // Sentence finished, add to history
            setHistory(prev => [{ id: Date.now(), text: phrase }, ...prev]);
          }
        }, 300); // 300ms per word to simulate real-time generation

      }, 1000); // 1 second processing delay
      
    }, 8000); // every 8 seconds start a new sentence

    return () => {
      clearInterval(simulationIntervalRef.current);
      clearInterval(wordIntervalRef.current);
    };
  }, [isDetecting]);

  const clearSession = () => {
    setCurrentText("");
    setHistory([]);
  };

  return { currentText, setCurrentText, status, history, clearSession };
}
