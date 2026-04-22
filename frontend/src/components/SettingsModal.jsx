import { X, Camera, Zap, ShieldAlert } from 'lucide-react';
import { useState, useEffect } from 'react';

export default function SettingsModal({ onClose }) {
  const [devices, setDevices] = useState([]);
  const [performanceMode, setPerformanceMode] = useState('low');

  useEffect(() => {
    // try to fetch camera devices
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      navigator.mediaDevices.enumerateDevices()
        .then(devs => {
          setDevices(devs.filter(d => d.kind === 'videoinput'));
        })
        .catch(console.error);
    }
  }, []);

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-in fade-in duration-200">
      <div className="bg-dark-800 border border-gray-700 rounded-2xl w-full max-w-md shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
        <div className="flex justify-between items-center p-5 border-b border-gray-700">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <SettingsIcon size={20} className="text-gray-400" /> System Settings
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
            <X size={24} />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Camera Selection */}
          <div>
            <label className="text-sm font-medium text-gray-400 mb-2 flex items-center gap-2">
              <Camera size={16} /> Select Camera
            </label>
            <select className="w-full bg-dark-900 border border-gray-700 rounded-xl p-3 text-white appearance-none focus:ring-2 focus:ring-primary-500 outline-none">
              {devices.length === 0 ? (
                <option>Default System Camera</option>
              ) : (
                devices.map((d, i) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label || `Camera ${i + 1}`}
                  </option>
                ))
              )}
            </select>
          </div>

          {/* Performance Mode */}
          <div>
            <label className="text-sm font-medium text-gray-400 mb-2 flex items-center gap-2">
              <Zap size={16} /> Performance Mode
            </label>
            <div className="grid grid-cols-2 gap-3">
              <button 
                onClick={() => setPerformanceMode('low')}
                className={`p-3 rounded-xl font-medium text-sm text-left relative overflow-hidden transition-all duration-200 ${performanceMode === 'low' ? 'bg-primary-500/10 border-2 border-primary-500 text-primary-400' : 'bg-dark-900 border border-gray-700 text-gray-400 hover:border-gray-600'}`}
              >
                {performanceMode === 'low' && <div className="absolute top-0 right-0 w-2 h-2 bg-primary-500 rounded-bl-lg"></div>}
                <div className="font-bold mb-1">Low Latency</div>
                <div className="text-xs opacity-80 font-normal">Prioritizes speed (faster WebSocket sync)</div>
              </button>
              
              <button 
                onClick={() => setPerformanceMode('high')}
                className={`p-3 rounded-xl font-medium text-sm text-left relative overflow-hidden transition-all duration-200 ${performanceMode === 'high' ? 'bg-primary-500/10 border-2 border-primary-500 text-primary-400' : 'bg-dark-900 border border-gray-700 text-gray-400 hover:border-gray-600'}`}
              >
                {performanceMode === 'high' && <div className="absolute top-0 right-0 w-2 h-2 bg-primary-500 rounded-bl-lg"></div>}
                <div className="font-bold mb-1">High Accuracy</div>
                <div className="text-xs opacity-80 font-normal">Waits for complete poses (better context)</div>
              </button>
            </div>
          </div>

          {/* Connection Info */}
          <div className="bg-dark-900/50 p-4 rounded-xl border border-green-500/20 flex gap-3 text-sm">
             <ShieldAlert size={20} className="text-green-500 flex-shrink-0 mt-0.5" />
             <div className="text-gray-300">
               <span className="font-semibold block text-green-400 mb-1">Backend Connected</span>
               Live prediction via WebSocket API at <code>ws://localhost:8000/ws/stream</code>. Translation fires automatically when a sign is completed.
             </div>
          </div>
        </div>
        
        <div className="p-5 border-t border-gray-700 flex justify-end gap-3 bg-dark-900/30">
          <button onClick={onClose} className="btn-secondary">Cancel</button>
          <button onClick={onClose} className="btn-primary">Save Changes</button>
        </div>
      </div>
    </div>
  );
}

// Inline settings icon definition to avoid missing imports
function SettingsIcon(props) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path>
      <circle cx="12" cy="12" r="3"></circle>
    </svg>
  );
}
