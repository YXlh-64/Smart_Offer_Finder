import { useState, useRef } from "react";
import { Mic, Send, Loader2, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { transcribeAudio } from "@/lib/api";

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading?: boolean;
}

export function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const handleSend = () => {
    if (message.trim() && !isLoading) {
      onSend(message.trim());
      setMessage("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Use webm format for better compatibility
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
        
        // Create blob from chunks
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
        
        if (audioBlob.size > 0) {
          setIsTranscribing(true);
          try {
            const text = await transcribeAudio(audioBlob);
            if (text.trim()) {
              setMessage(prev => prev ? `${prev} ${text}` : text);
            }
          } catch (error) {
            console.error('Transcription error:', error);
          } finally {
            setIsTranscribing(false);
          }
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Impossible d\'accéder au microphone. Vérifiez les permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="p-4 border-t border-border bg-card">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-end gap-3 bg-secondary/50 rounded-2xl p-2 input-glow transition-shadow border border-border">
          {/* Text Input */}
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isTranscribing ? "Transcription en cours..." : "Posez votre question..."}
            rows={1}
            disabled={isTranscribing}
            className="flex-1 bg-transparent border-0 resize-none text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-0 py-2.5 max-h-32 min-h-[40px] disabled:opacity-50"
            style={{ height: "40px" }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = "40px";
              target.style.height = Math.min(target.scrollHeight, 128) + "px";
            }}
          />

          {/* Voice Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleRecording}
            disabled={isTranscribing || isLoading}
            className={cn(
              "flex-shrink-0 h-10 w-10 rounded-xl transition-colors",
              isRecording 
                ? "bg-destructive text-destructive-foreground hover:bg-destructive/90" 
                : isTranscribing
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:text-foreground hover:bg-secondary"
            )}
          >
            {isTranscribing ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : isRecording ? (
              <Square className="w-4 h-4 fill-current" />
            ) : (
              <Mic className="w-5 h-5" />
            )}
          </Button>

          {/* Send Button */}
          <Button
            onClick={handleSend}
            disabled={!message.trim() || isLoading || isTranscribing}
            className="flex-shrink-0 h-10 w-10 rounded-xl chat-gradient hover:opacity-90 disabled:opacity-50"
            size="icon"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </Button>
        </div>

        <p className="text-xs text-muted-foreground text-center mt-3">
          FORSA Assistant peut faire des erreurs. Vérifiez les informations importantes.
        </p>
      </div>
    </div>
  );
}
