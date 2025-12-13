import { useRef, useEffect } from "react";
import { MessageBubble } from "./MessageBubble";
import { ChatInput } from "./ChatInput";
import { Sparkles } from "lucide-react";

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  citations?: { title: string; type?: string }[];
  timestamp?: string;
  isStreaming?: boolean;
}

interface ChatAreaProps {
  messages: Message[];
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
}

export function ChatArea({ messages, onSendMessage, isLoading }: ChatAreaProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Messages Area */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto scrollbar-thin p-6"
      >
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
              <div className="w-16 h-16 rounded-2xl chat-gradient flex items-center justify-center mb-6 shadow-medium">
                <Sparkles className="w-8 h-8 text-primary-foreground" />
              </div>
              <h2 className="text-2xl font-semibold text-foreground mb-2">
                Bienvenue sur FORSA Assistant
              </h2>
              <p className="text-muted-foreground max-w-md">
                Je suis votre assistant intelligent pour les offres, conventions et procédures d'Algérie Télécom. Comment puis-je vous aider ?
              </p>
              <div className="flex flex-wrap gap-2 mt-6 justify-center">
                {[
                  "Quelles sont les offres Idoom Fibre ?",
                  "Détails convention Huawei",
                  "Procédure NGBSS abonnement"
                ].map((suggestion, i) => (
                  <button
                    key={i}
                    onClick={() => onSendMessage(suggestion)}
                    className="px-4 py-2 bg-card border border-border rounded-full text-sm text-foreground hover:bg-secondary transition-colors shadow-soft"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <MessageBubble
                key={message.id}
                content={message.content}
                isUser={message.isUser}
                citations={message.citations}
                timestamp={message.timestamp}
                isStreaming={message.isStreaming}
              />
            ))
          )}
        </div>
      </div>

      {/* Input Area */}
      <ChatInput onSend={onSendMessage} isLoading={isLoading} />
    </div>
  );
}
