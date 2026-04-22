import { useRef, useEffect, useState } from 'react';
import { Camera, CameraOff } from 'lucide-react';

export default function VideoFeed({ status, isDetecting, sendFrame, currentText }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [hasCamera, setHasCamera] = useState(false);

  useEffect(() => {
    let stream = null;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setHasCamera(true);
      } catch (err) {
        console.error("Error accessing camera:", err);
        setHasCamera(false);
      }
    };

    startCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Frame extraction loop
  useEffect(() => {
    let interval;
    if (isDetecting && hasCamera && videoRef.current) {
      interval = setInterval(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (video.videoWidth > 0 && canvas) {
          canvas.width = 960;  // Higher resolution for better keypoint detection
          canvas.height = 720;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const base64Jpeg = canvas.toDataURL('image/jpeg', 0.9); // 90% quality to preserve hand detail
          sendFrame(base64Jpeg);
        }
      }, 80); // ~12.5 FPS — closer to typical sign language dataset frame rate
    }
    
    return () => clearInterval(interval);
  }, [isDetecting, hasCamera, sendFrame]);

  return (
    <div className="flex-1 w-full h-full relative bg-[#0B0F19] rounded-[2rem] overflow-hidden flex items-center justify-center border border-white/5">
      {/* Hidden canvas for frame extraction */}
      <canvas ref={canvasRef} className="hidden" />
      {hasCamera ? (
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          className={`w-full h-full object-cover -scale-x-100 transition-opacity duration-500 ${isDetecting ? 'opacity-100' : 'opacity-20 blur-md grayscale'}`} 
        />
      ) : (
        <div className="flex flex-col items-center justify-center text-gray-500">
          <CameraOff size={48} className="mb-4 opacity-50" />
          <p>Camera access denied or unavailable</p>
        </div>
      )}

      {/* Hand Detection Overlay Layer */}
      {isDetecting && hasCamera && (
        <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center">
           <div className="w-72 h-72 border border-primary-500/30 rounded-2xl relative">
              {/* Subtle landmark points */}
              <div className="absolute top-4 left-4 w-1.5 h-1.5 bg-primary-400 rounded-full shadow-[0_0_8px_rgba(56,189,248,0.8)]"></div>
              <div className="absolute bottom-8 right-12 w-1.5 h-1.5 bg-primary-400 rounded-full shadow-[0_0_8px_rgba(56,189,248,0.8)]"></div>
              <div className="absolute top-1/2 left-1/3 w-1 h-1 bg-accent-400 rounded-full"></div>
              
              {/* Corner brackets */}
              <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-primary-500/50 rounded-tl-lg"></div>
              <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-primary-500/50 rounded-tr-lg"></div>
              <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-primary-500/50 rounded-bl-lg"></div>
              <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-primary-500/50 rounded-br-lg"></div>
           </div>
           
           {/* Guidance feedback */}
           <div className="mt-8 text-primary-400/80 text-sm font-medium tracking-wide animate-pulse-slow">
             {status.includes('unclear') ? 'Show your hands clearly...' : 'Tracking landmarks...'}
           </div>
        </div>
      )}

      {/* Cinematic Subtitles */}
      <div className={`absolute bottom-28 md:bottom-12 left-0 right-0 px-4 md:px-8 flex justify-center pointer-events-none transition-all duration-500 ${isDetecting && currentText ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
        <div className="bg-gradient-to-t from-black/90 via-black/40 to-transparent pt-12 pb-6 px-8 w-full max-w-4xl text-center rounded-2xl backdrop-blur-sm">
          <p className="text-3xl md:text-5xl font-display font-bold text-white drop-shadow-[0_4px_16px_rgba(0,0,0,0.9)] tracking-wide leading-tight">
            {currentText}
          </p>
        </div>
      </div>
    </div>
  );
}
