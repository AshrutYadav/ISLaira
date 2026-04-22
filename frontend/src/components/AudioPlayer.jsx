import { useState, useEffect } from 'react';
import { Volume2, VolumeX, PlayCircle } from 'lucide-react';

export default function AudioPlayer({ textToPlay }) {
  const [autoPlay, setAutoPlay] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);

  // When text finishes generating, play it
  useEffect(() => {
    if (autoPlay && textToPlay && !isPlaying) {
      // Small delay just to let the eye catch the text
      const timeout = setTimeout(() => {
        playSpeech(textToPlay);
      }, 500);
      return () => clearTimeout(timeout);
    }
  }, [textToPlay, autoPlay]);

  // Enter hotkey to toggle Auto-Play
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Enter') {
        e.preventDefault();
        setAutoPlay((prev) => {
          const next = !prev;
          if (!next && window.speechSynthesis) {
            window.speechSynthesis.cancel();
          }
          return next;
        });
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const playSpeech = (text) => {
    if (!text || !window.speechSynthesis) return;

    window.speechSynthesis.cancel(); // kill active speech

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => setIsPlaying(true);
    utterance.onend = () => setIsPlaying(false);
    utterance.onerror = () => setIsPlaying(false);

    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="flex items-center justify-between bg-dark-800 p-3 rounded-xl border border-gray-700/50">
      <div className="flex items-center gap-3 w-full">
        <button 
          onClick={() => playSpeech(textToPlay)}
          disabled={!textToPlay || isPlaying}
          className={`p-2 rounded-full transition-colors flex-shrink-0 ${isPlaying ? 'bg-primary-500/20 text-primary-400' : 'bg-dark-700 hover:bg-dark-600 text-gray-300'} disabled:opacity-50`}
        >
          {isPlaying ? (
            <div className="flex gap-1 items-center justify-center p-1 px-1.5 h-6">
              <div className="w-1 h-3 bg-primary-400 animate-[bounce_1s_infinite_0ms] rounded-full"></div>
              <div className="w-1 h-4 bg-primary-400 animate-[bounce_1s_infinite_200ms] rounded-full"></div>
              <div className="w-1 h-2 bg-primary-400 animate-[bounce_1s_infinite_400ms] rounded-full"></div>
            </div>
          ) : (
             <PlayCircle size={24} />
          )}
        </button>

        <div className="flex-1 text-sm text-gray-400 truncate pr-4">
          {isPlaying ? "Speaking..." : (textToPlay ? "Ready to play" : "Audio synthesis ready")}
        </div>

        <button 
          onClick={() => {
            setAutoPlay(!autoPlay);
            if(isPlaying) window.speechSynthesis.cancel();
          }}
          className={`flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-full transition-colors ${autoPlay ? 'bg-primary-500/10 text-primary-400 border border-primary-500/20' : 'bg-dark-700 text-gray-400 border border-transparent hover:bg-dark-600'}`}
        >
          {autoPlay ? <Volume2 size={14} /> : <VolumeX size={14} />}
          {autoPlay ? 'Auto-Play ON' : 'Auto-Play OFF'}
        </button>
      </div>
    </div>
  );
}
