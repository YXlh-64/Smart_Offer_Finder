import { useState, useCallback } from 'react';

interface UseSmartChatProps {
  sessionId: string;
}

export function useSmartChat({ sessionId }: UseSmartChatProps) {
  const [isConnected, setIsConnected] = useState<boolean | null>(null);

  const checkConnection = useCallback(async () => {
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        setIsConnected(true);
      } else {
        setIsConnected(false);
      }
    } catch (error) {
      console.error('Connection check failed:', error);
      setIsConnected(false);
    }
  }, []);

  return {
    isConnected,
    checkConnection,
    sessionId,
  };
}
