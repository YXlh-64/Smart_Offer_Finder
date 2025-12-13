import { useState, useEffect, useCallback } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { ChatArea } from "@/components/chat/ChatArea";
import { useSmartChat } from "@/hooks/use-smart-chat";
import { useToast } from "@/hooks/use-toast";
import { 
  getSessions, 
  getSession, 
  createSession, 
  deleteSession,
  Session 
} from "@/lib/api";

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  citations?: { title: string; type?: string; filePath?: string }[];
  timestamp?: string;
  isStreaming?: boolean;
}

const Index = () => {
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();
  
  const { 
    isConnected,
    checkConnection 
  } = useSmartChat({
    sessionId: activeSessionId || "default",
  });

  // Load sessions from backend
  const loadSessions = useCallback(async () => {
    try {
      setSessionsLoading(true);
      const fetchedSessions = await getSessions();
      setSessions(fetchedSessions);
    } catch (error) {
      console.error("Failed to load sessions:", error);
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  // Check backend connection and load sessions on mount
  useEffect(() => {
    checkConnection();
    loadSessions();
  }, [checkConnection, loadSessions]);

  // Show connection status toast
  useEffect(() => {
    if (isConnected === false) {
      toast({
        title: "Backend non connecté",
        description: "Assurez-vous que le serveur FastAPI est démarré sur le port 8001",
        variant: "destructive",
      });
    } else if (isConnected === true) {
      toast({
        title: "Connecté",
        description: "Le système est prêt à répondre à vos questions",
      });
    }
  }, [isConnected, toast]);

  // Create a new chat session
  const handleNewChat = async () => {
    try {
      const newSession = await createSession();
      setSessions((prev) => [newSession, ...prev]);
      setActiveSessionId(newSession.id);
      setMessages([]);
    } catch (error) {
      console.error("Failed to create session:", error);
      // Fallback to local session if backend fails
      const fallbackId = `local-${Date.now()}`;
      setActiveSessionId(fallbackId);
      setMessages([]);
    }
  };

  // Select an existing session
  const handleSelectSession = async (sessionId: string) => {
    try {
      setActiveSessionId(sessionId);
      const sessionData = await getSession(sessionId);
      
      // Convert session messages to our Message format
      const loadedMessages: Message[] = sessionData.messages.map((msg, index) => ({
        id: `msg-${index}-${msg.timestamp}`,
        content: msg.content,
        isUser: msg.role === "user",
        citations: msg.sources?.map((s) => ({ title: s.split(/[/\\]/).pop() || s })),
        timestamp: new Date(msg.timestamp).toLocaleTimeString("fr-FR", { 
          hour: "2-digit", 
          minute: "2-digit" 
        }),
      }));
      
      setMessages(loadedMessages);
    } catch (error) {
      console.error("Failed to load session:", error);
      toast({
        title: "Erreur",
        description: "Impossible de charger la conversation",
        variant: "destructive",
      });
    }
  };

  // Delete a session
  const handleDeleteSession = async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      
      // If we deleted the active session, clear the chat
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
      }
      
      toast({
        title: "Supprimé",
        description: "La conversation a été supprimée",
      });
    } catch (error) {
      console.error("Failed to delete session:", error);
      toast({
        title: "Erreur",
        description: "Impossible de supprimer la conversation",
        variant: "destructive",
      });
    }
  };

  const handleSendMessage = async (content: string) => {
    // If no active session, create one first
    let currentSessionId = activeSessionId;
    if (!currentSessionId) {
      try {
        const newSession = await createSession(content.slice(0, 40));
        setSessions((prev) => [newSession, ...prev]);
        currentSessionId = newSession.id;
        setActiveSessionId(newSession.id);
      } catch (error) {
        console.error("Failed to create session:", error);
        currentSessionId = `local-${Date.now()}`;
        setActiveSessionId(currentSessionId);
      }
    }

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      content,
      isUser: true,
      timestamp: new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" }),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    // Add placeholder for AI response
    const aiMessageId = `ai-${Date.now()}`;
    const aiMessage: Message = {
      id: aiMessageId,
      content: '',
      isUser: false,
      timestamp: new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" }),
      isStreaming: true,
    };
    setMessages((prev) => [...prev, aiMessage]);

    try {
      const { sendChatMessageStream } = await import('@/lib/api');
      await sendChatMessageStream(
        { question: content, session_id: currentSessionId },
        // On chunk - update the AI message content
        (text: string) => {
          setMessages((prev) => 
            prev.map((msg) => 
              msg.id === aiMessageId 
                ? { ...msg, content: text }
                : msg
            )
          );
        },
        // On sources - update citations with file paths for download
        (sources: string[]) => {
          const citations = sources.map((source) => {
            const filename = source.split(/[/\\]/).pop() || source;
            return { title: filename, filePath: source };
          });
          setMessages((prev) => 
            prev.map((msg) => 
              msg.id === aiMessageId 
                ? { ...msg, citations }
                : msg
            )
          );
        },
        // On complete
        () => {
          setMessages((prev) => 
            prev.map((msg) => 
              msg.id === aiMessageId 
                ? { ...msg, isStreaming: false }
                : msg
            )
          );
          setIsLoading(false);
          // Refresh sessions to get updated titles
          loadSessions();
        },
        // On error
        (error: string) => {
          setMessages((prev) => 
            prev.map((msg) => 
              msg.id === aiMessageId 
                ? { ...msg, content: `Erreur: ${error}`, isStreaming: false }
                : msg
            )
          );
          setIsLoading(false);
          toast({
            title: "Erreur",
            description: error,
            variant: "destructive",
          });
        }
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Erreur inconnue';
      setMessages((prev) => 
        prev.map((msg) => 
          msg.id === aiMessageId 
            ? { ...msg, content: `Erreur de connexion: ${errorMessage}`, isStreaming: false }
            : msg
        )
      );
      setIsLoading(false);
      toast({
        title: "Erreur",
        description: errorMessage,
        variant: "destructive",
      });
    }
  };

  return (
    <div className="flex h-screen w-full overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        sessions={sessions}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        activeSessionId={activeSessionId || undefined}
        isLoading={sessionsLoading}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <ChatArea
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
};

export default Index;
