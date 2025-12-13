# Frontend Setup

A React + TypeScript chatbot UI for the Smart Offer Finder RAG system built with Vite and Tailwind CSS.

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components
- **TanStack Query** - Data fetching
- **React Markdown** - Markdown rendering

## Installation

```bash
cd frontend
npm install
```

## Running the UI

1. Make sure the FastAPI backend is running on `http://localhost:8000`:
   ```bash
   # From the project root
   python main.py
   ```

2. Start the frontend development server:
   ```bash
   npm run dev
   ```

3. Open `http://localhost:8080` in your browser.

## Features

- Modern, responsive chat interface
- Real-time message streaming with SSE
- Source citations from retrieved documents
- Markdown support in responses
- Connection status indicator
- Chat history clearing
- Error handling with toast notifications

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Available variables:
- `VITE_API_URL` - Backend API URL (default: `http://localhost:8000`)

### Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/          # shadcn/ui components
│   │   └── ChatInterface.tsx
│   ├── services/
│   │   └── api.ts       # Backend API client
│   ├── lib/
│   │   └── utils.ts     # Utility functions
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── index.html
├── vite.config.ts
├── tailwind.config.js
└── package.json
```
