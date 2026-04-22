import { Play, Square, RefreshCcw, Trash2, Globe } from 'lucide-react';

export default function ControlBar({ isDetecting, setIsDetecting, clearSession }) {
  return (
    <div className={`flex items-center gap-1 md:gap-2 p-2 bg-[#141B2D]/80 backdrop-blur-2xl border transition-all duration-500 rounded-full shadow-[0_8px_32px_rgba(0,0,0,0.6)] ${isDetecting ? 'border-primary-500/30 shadow-[0_0_24px_rgba(14,165,233,0.15)]' : 'border-white/10'}`}>
      
      {isDetecting ? (
        <button 
          onClick={() => setIsDetecting(false)}
          className="flex items-center gap-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 font-semibold py-3 px-6 rounded-full transition-all duration-300 active:scale-95 group relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-red-500/10 animate-pulse"></div>
          <Square size={20} className="fill-current relative z-10" /> 
          <span className="relative z-10 tracking-wide">Stop Detection</span>
        </button>
      ) : (
        <button 
          onClick={() => setIsDetecting(true)}
          className="flex items-center gap-2 bg-gradient-to-r from-primary-500 to-accent-500 hover:from-primary-400 hover:to-accent-400 text-white font-semibold py-3 px-6 rounded-full transition-all duration-300 active:scale-95 shadow-lg shadow-primary-500/20"
        >
          <Play size={20} className="fill-current" /> 
          <span className="tracking-wide">Start Detection</span>
        </button>
      )}

      <div className="w-px h-8 bg-white/10 mx-1 md:mx-2"></div>

      <button 
        onClick={clearSession}
        className="flex items-center gap-2 px-4 py-3 rounded-full hover:bg-white/5 text-gray-300 hover:text-white transition-colors text-sm font-medium"
      >
        <Trash2 size={18} className="text-gray-400" /> 
        <span className="hidden sm:inline">Clear Log</span>
      </button>

      <button 
        onClick={clearSession}
        className="flex items-center gap-2 px-4 py-3 rounded-full hover:bg-white/5 text-gray-300 hover:text-white transition-colors text-sm font-medium"
      >
        <RefreshCcw size={18} className="text-gray-400" />
        <span className="hidden sm:inline">Reset Frame</span>
      </button>

      <div className="w-px h-8 bg-white/10 mx-1 md:mx-2 hidden md:block"></div>

      <div className="hidden md:flex items-center gap-2 px-4 py-2 bg-[#0B0F19]/60 rounded-full border border-white/5">
        <Globe size={18} className="text-primary-400" />
        <select className="bg-transparent text-sm font-medium text-gray-300 outline-none cursor-pointer">
          <option value="en">English Output</option>
          <option value="hi" disabled>Hindi (Beta)</option>
        </select>
      </div>
    </div>
  );
}
