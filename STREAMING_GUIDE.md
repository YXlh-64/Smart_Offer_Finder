# Streaming Implementation Guide

This document explains the streaming implementation for the Smart Offer Finder application.

## Overview

The application now supports real-time streaming of chat responses using Server-Sent Events (SSE). This provides a better user experience with responses appearing incrementally rather than all at once.

## Architecture

### Backend (FastAPI - `main.py`)

The FastAPI server provides two endpoints:

1. **`POST /chat`** - Traditional non-streaming endpoint for backward compatibility
   - Sends complete response with all sources at once
   - Useful for integrations that don't support streaming

2. **`POST /chat/stream`** - Streaming endpoint (recommended)
   - Uses Server-Sent Events (SSE) for streaming
   - Sends response in chunks as it's being generated
   - Sends sources separately after the complete answer

### Frontend (React - `ChatApp.jsx`)

The React component now:
1. Sends requests to the `/chat/stream` endpoint
2. Uses the Fetch API with `response.body.getReader()` to read streaming data
3. Parses JSON objects from each line of the response
4. Updates the UI in real-time as chunks arrive
5. Displays sources once the response is complete

## Response Format (SSE)

The streaming endpoint sends JSON-formatted lines of text:

```json
{"type": "chunk", "content": "partial response text..."}
```

- **chunk**: Partial response content that accumulates
- **sources**: Array of source documents
- **complete**: Signals end of response
- **error**: Error message if something went wrong

Example streaming sequence:
```
{"type": "chunk", "content": "The available offers include..."}
{"type": "chunk", "content": "The available offers include business plans, voice..."}
{"type": "chunk", "content": "The available offers include business plans, voice, and data packages."}
{"type": "sources", "content": ["doc1.pdf", "doc2.pdf"]}
{"type": "complete"}
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key additions:
- `fastapi>=0.104.0` - Web framework with streaming support
- `uvicorn>=0.24.0` - ASGI server for running FastAPI

### 2. Run the Backend

Start the FastAPI server:

```bash
python main.py
```

The server will start on `http://localhost:8000`

Check health status:
```bash
curl http://localhost:8000/health
```

### 3. Run the Frontend

In the `frontend` directory:

```bash
npm install
npm start
```

The React app will start on `http://localhost:3000`

## Testing the Streaming

### Using cURL

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Les retraités de l’établissement B peuvent-ils bénéficier d'un deuxième accès ?", "session_id": "default"}'
```

### Using Python

```python
import requests
import json

url = "http://localhost:8000/chat/stream"
payload = {
    "question": "What are the available offers?",
    "session_id": "default"
}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        data = json.loads(line)
        if data['type'] == 'chunk':
            print(data['content'], end='', flush=True)
        elif data['type'] == 'sources':
            print(f"\n\nSources: {data['content']}")
```

## How Streaming Works

1. **Request**: User sends a question via the frontend
2. **Response Stream**: Backend starts streaming the response in chunks
3. **Real-time Update**: Frontend receives each chunk and updates the UI
4. **Sources**: After the answer, sources are sent as metadata
5. **Completion**: A final "complete" message signals end of stream

## Customizing Streaming Speed

In `main.py`, the `stream_chat_response()` function controls streaming:

```python
# Currently splits response into chunks every 5 words
# Modify this to control streaming speed:
if (i + 1) % 5 == 0 or i == len(words) - 1:
    yield json.dumps({"type": "chunk", "content": accumulated_text}) + "\n"
    accumulated_text = ""
    await asyncio.sleep(0.01)  # Adjust this delay
```

**Options:**
- Decrease the modulo (e.g., `% 3`) for finer granularity
- Increase `asyncio.sleep()` for slower streaming
- Decrease it for faster streaming

## Advantages of Streaming

✅ **Better UX**: Users see responses appearing in real-time  
✅ **Feedback**: Visual feedback that something is happening  
✅ **Scalability**: Can handle longer responses without timeout  
✅ **Interactivity**: Users can start reading before full response arrives  
✅ **Latency Perception**: Feels faster due to incremental display  

## Backward Compatibility

The non-streaming `/chat` endpoint is still available for clients that don't support streaming:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the available offers?", "session_id": "default"}'
```

## Troubleshooting

### Streaming not working in browser
- Check CORS settings in FastAPI (already configured in main.py)
- Verify fetch API is supported (modern browsers only)
- Check browser console for errors

### Connection timeout
- Ensure FastAPI server is running: `python main.py`
- Check if port 8000 is available
- Verify no firewall blocking connections

### Incomplete responses
- Check backend logs for errors
- Ensure ChromaDB is properly initialized
- Verify documents have been ingested

## Files Modified

- **`main.py`** - New FastAPI application with streaming support
- **`requirements.txt`** - Added fastapi and uvicorn
- **`frontend/src/ChatApp.jsx`** - Updated to use streaming endpoint with fetch API

## API Documentation

Once FastAPI is running, visit:
```
http://localhost:8000/docs
```

This provides interactive Swagger documentation for all endpoints.
