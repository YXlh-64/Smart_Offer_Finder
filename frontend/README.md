# Frontend Setup

A React-based chatbot UI for the Smart Offer Finder RAG system.

## Installation

```bash
cd frontend
npm install
```

## Running the UI

Make sure the FastAPI backend is running on `http://localhost:8000`, then:

```bash
npm start
```

The UI will open at `http://localhost:3000`.

## Features

- Clean, modern chat interface
- Real-time message streaming
- Source citations from retrieved documents
- Responsive design
- Typing indicators
- Chat history clearing
- Error handling with user feedback

## Configuration

If your FastAPI server runs on a different port, update `API_BASE_URL` in `src/ChatApp.jsx`:

```javascript
const API_BASE_URL = 'http://localhost:8000'; // Change this if needed
```

For production CORS setup, update the FastAPI server's CORS middleware in `src/chat.py`.
