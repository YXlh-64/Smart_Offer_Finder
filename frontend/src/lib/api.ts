interface ChatMessage {
  question: string;
  session_id: string;
}

interface StreamChunk {
  type: 'chunk' | 'sources' | 'complete' | 'error';
  content?: string | string[];
}

/**
 * Send a chat message with streaming response
 */
export async function sendChatMessageStream(
  message: ChatMessage,
  onChunk: (text: string) => void,
  onSources: (sources: string[]) => void,
  onComplete: () => void,
  onError: (error: string) => void
): Promise<void> {
  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(message),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let accumulatedText = '';

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;

      // Decode the chunk and add to buffer
      buffer += decoder.decode(value, { stream: true });
      
      // Process complete JSON lines
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (!line.trim()) continue;

        try {
          const data: StreamChunk = JSON.parse(line);

          switch (data.type) {
            case 'chunk':
              if (typeof data.content === 'string') {
                // Append chunks - backend sends incremental pieces
                accumulatedText += data.content;
                onChunk(accumulatedText);
              }
              break;

            case 'sources':
              if (Array.isArray(data.content)) {
                onSources(data.content);
              }
              break;

            case 'complete':
              onComplete();
              return;

            case 'error':
              if (typeof data.content === 'string') {
                onError(data.content);
              }
              return;
          }
        } catch (e) {
          console.error('Failed to parse JSON line:', line, e);
        }
      }
    }

    // Call complete if we finished without explicit complete signal
    onComplete();
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    onError(errorMessage);
  }
}

/**
 * Non-streaming chat endpoint for backward compatibility
 */
export async function sendChatMessage(message: ChatMessage): Promise<{
  answer: string;
  sources: string[];
}> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(message),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

/**
 * Transcribe audio to text using Whisper
 */
export async function transcribeAudio(audioBlob: Blob): Promise<string> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');

  const response = await fetch('/api/transcribe', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Transcription failed' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  const result = await response.json();
  return result.text;
}

/**
 * Check backend health status
 */
export async function checkHealth(): Promise<{
  status: string;
  chain_initialized: boolean;
}> {
  const response = await fetch('/api/health');
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

// ============== Session Management ==============

export interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface SessionMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: string[];
}

export interface SessionWithMessages extends Session {
  messages: SessionMessage[];
}

/**
 * Get all chat sessions
 */
export async function getSessions(): Promise<Session[]> {
  const response = await fetch('/api/sessions');
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  return data.sessions;
}

/**
 * Create a new chat session
 */
export async function createSession(title?: string): Promise<Session> {
  const response = await fetch('/api/sessions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title: title || 'Nouvelle conversation' }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

/**
 * Get a specific session with all messages
 */
export async function getSession(sessionId: string): Promise<SessionWithMessages> {
  const response = await fetch(`/api/sessions/${sessionId}`);
  
  if (!response.ok) {
    if (response.status === 404) {
      throw new Error('Session not found');
    }
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

/**
 * Update a session's title
 */
export async function updateSessionTitle(sessionId: string, title: string): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
}

/**
 * Delete a chat session
 */
export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
}
