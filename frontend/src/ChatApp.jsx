import React, { useState, useRef, useEffect } from 'react';
import './ChatApp.css';

const ChatApp = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const API_BASE_URL = 'http://localhost:8000';

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();

    if (!input.trim()) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: input,
    };
    const currentInput = input;
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Create streaming bot message placeholder
      const botMessageId = Date.now() + 1;
      const botMessage = {
        id: botMessageId,
        type: 'bot',
        text: '',
        sources: [],
      };
      setMessages((prev) => [...prev, botMessage]);

      // Call FastAPI streaming endpoint
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: currentInput,
          session_id: 'default',
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let streamedText = '';
      let sources = [];

      // Read streaming response
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter((line) => line.trim());

        for (const line of lines) {
          try {
            const data = JSON.parse(line);

            if (data.type === 'chunk') {
              // Update streamed text
              streamedText = data.content;
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === botMessageId
                    ? { ...msg, text: streamedText }
                    : msg
                )
              );
            } else if (data.type === 'sources') {
              // Add sources
              sources = data.content;
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === botMessageId
                    ? { ...msg, sources: sources }
                    : msg
                )
              );
            } else if (data.type === 'error') {
              // Handle error
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === botMessageId
                    ? { ...msg, text: `Error: ${data.content}` }
                    : msg
                )
              );
              break;
            }
          } catch (parseError) {
            // Ignore JSON parse errors
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        text: 'Sorry, an error occurred while processing your request. Make sure the FastAPI server is running on http://localhost:8000.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setInput('');
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>🎯 Smart Offer Finder</h1>
        <p>Intelligent Assistant for Algeria Telecom Offers & Conventions</p>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h2>Welcome!</h2>
            <p>Ask me anything about Algeria Telecom offers, interconnection services, or conventions.</p>
            <p style={{ marginTop: '10px', fontSize: '0.9em', color: '#999' }}>
              Example: "What interconnection services are available?"
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`message message-${msg.type}`}>
            <div className="message-content">
              <p>{msg.text}</p>
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  <strong>Sources:</strong>
                  <ul>
                    {msg.sources.map((source, idx) => (
                      <li key={idx}>{source}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message message-bot">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={sendMessage} className="chat-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your question here..."
          disabled={loading}
          className="chat-input"
        />
        <button type="submit" disabled={loading} className="send-button">
          {loading ? '⏳' : '📤'}
        </button>
        <button
          type="button"
          onClick={clearChat}
          className="clear-button"
          title="Clear chat history"
        >
          🗑️
        </button>
      </form>
    </div>
  );
};

export default ChatApp;
