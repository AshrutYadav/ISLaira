import { useState, useEffect } from 'react';
import VideoFeed from './components/VideoFeed';
import OutputPanel from './components/OutputPanel';
import ControlBar from './components/ControlBar';
import SettingsModal from './components/SettingsModal';
import { useRealTimeConnection } from './hooks/useRealTimeConnection';
import { Settings } from 'lucide-react';

function App() {
  const [isDetecting, setIsDetecting] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const { currentText, setCurrentText, status, history, liveGesture, clearSession, sendFrame } = useRealTimeConnection(isDetecting);

  // Spacebar hotkey to toggle detection
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space') {
        e.preventDefault(); // Prevent scrolling
        setIsDetecting(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="h-screen w-full bg-dark-900 relative overflow-hidden flex items-center justify-center">
      {/* Background Video Layer */}
      <div className="absolute inset-0 md:p-4">
        <VideoFeed status={status} isDetecting={isDetecting} sendFrame={sendFrame} currentText={currentText} />
      </div>

      {/* Foreground UI Layer - Header */}
      <header className="absolute top-6 left-6 right-6 flex justify-between items-start pointer-events-none z-20">
        <div className="flex flex-col gap-3 pointer-events-auto">
          {/* Main Title Box */}
          <div className="bg-[#0B0F19]/80 backdrop-blur-md px-5 py-3 rounded-2xl border border-white/10 shadow-lg inline-block">
            <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-primary-400 to-accent-400 bg-clip-text text-transparent">
              VocalBridge
            </h1>
            <p className="text-gray-300 text-xs md:text-sm font-medium">Real-Time Sign Language Translator</p>
          </div>
          
          {/* Status Indicator (Moved here from VideoFeed to avoid overlap) */}
          <div className="inline-flex items-center gap-3 bg-[#141B2D]/80 backdrop-blur-md border border-white/10 px-4 py-2 rounded-full text-xs font-semibold tracking-wider text-gray-200 shadow-xl self-start">
            <span className="relative flex h-2.5 w-2.5">
              {status.includes("Detecting") || status.includes("Processing") ? (
                 <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${status.includes("Detecting") ? 'bg-accent-400' : 'bg-yellow-400'}`}></span>
              ) : null}
              <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${status.includes("Detecting") ? 'bg-accent-500' : status.includes("Processing") ? 'bg-yellow-500' : status.includes("unclear") ? 'bg-orange-500' : 'bg-red-500'}`}></span>
            </span>
            {status.toUpperCase()}
          </div>
        </div>
        <button className="btn-icon bg-dark-900/60 backdrop-blur-md pointer-events-auto hover:bg-dark-700/80 rounded-full p-3 border border-white/10 shadow-lg" onClick={() => setShowSettings(true)}>
          <Settings size={24} className="text-white" />
        </button>
      </header>

      {/* Right Sidebar - Output Panel */}
      <div className="absolute right-6 top-28 bottom-28 w-80 lg:w-96 pointer-events-none z-20 hidden md:flex flex-col">
        <div className="flex-1 pointer-events-auto w-full">
           <OutputPanel history={history} liveGesture={liveGesture} />
        </div>
      </div>

      {/* Floating Control Bar at Bottom Center */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 pointer-events-none z-30 w-full px-4 flex justify-center">
        <div className="pointer-events-auto">
          <ControlBar 
            isDetecting={isDetecting} 
            setIsDetecting={setIsDetecting} 
            clearSession={clearSession} 
          />
        </div>
      </div>

      {/* Modals */}
      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
    </div>
  );
}

export default App;
