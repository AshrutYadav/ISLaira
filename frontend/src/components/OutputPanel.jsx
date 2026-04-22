import { MessageSquare, Clock, Hand } from 'lucide-react';
import AudioPlayer from './AudioPlayer';

export default function OutputPanel({ history, liveGesture }) {
  return (
    <div className="flex flex-col h-full overflow-hidden bg-[#0B0F19]/80 backdrop-blur-xl border border-white/10 shadow-2xl rounded-[2rem] p-5">
      
      {/* Header & Audio */}
      <div className="flex-shrink-0 mb-4 flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-white/5">
        <h2 className="text-sm font-display uppercase tracking-wider font-semibold text-gray-400 flex items-center gap-2">
          <Clock size={16} className="text-primary-400" /> Translation Log
        </h2>
        <AudioPlayer textToPlay={history?.length > 0 ? history[0].text : ""} />
      </div>

      {/* Live Gesture Indicator */}
      {liveGesture && (
        <div className="mb-4 flex-shrink-0 bg-[#141B2D] border border-primary-500/20 rounded-2xl p-3 shadow-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-accent-500"></span>
              </span>
              <span className="text-xs font-semibold uppercase tracking-wider text-gray-300">
                Detecting
              </span>
            </div>
            <span className="text-[10px] font-mono bg-dark-900/80 px-2 py-0.5 rounded-full text-accent-400 border border-accent-500/20">
              {liveGesture.confidence}%
            </span>
          </div>
          <p className="mt-2 text-white font-medium text-sm truncate">
             <span className="text-primary-300">{liveGesture.gesture}</span>
            <span className="text-gray-400 mx-2">→</span> 
            {liveGesture.phrase}
          </p>
          {/* Confidence bar */}
          <div className="mt-2 h-1 bg-[#0B0F19] rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary-500 to-accent-500 transition-all duration-300"
              style={{ width: `${liveGesture.confidence}%` }}
            />
          </div>
        </div>
      )}

      {/* History Feed */}
      <div className="flex-1 overflow-y-auto flex flex-col gap-3 pr-1">
        {!history || history.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 opacity-60">
            <MessageSquare size={32} className="mb-3" />
            <p className="text-sm font-medium">No translations yet</p>
          </div>
        ) : (
          history.map((item, index) => (
            <div 
              key={item.id} 
              className={`p-3.5 rounded-2xl border transition-all duration-300 ${
                index === 0 
                  ? 'bg-primary-500/10 border-primary-500/20 shadow-md' 
                  : 'bg-[#141B2D]/60 border-white/5'
              }`}
            >
              <p className={`text-sm ${index === 0 ? 'text-white font-medium' : 'text-gray-300'}`}>
                {item.text}
              </p>
              <div className="flex items-center justify-between mt-2">
                {item.gesture && (
                  <span className="text-[10px] uppercase font-semibold text-primary-400 bg-primary-500/10 px-1.5 py-0.5 rounded">
                    {item.gesture}
                  </span>
                )}
                <span className="text-[10px] text-gray-500">
                  {new Date(item.id).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
