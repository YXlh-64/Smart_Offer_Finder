#!/usr/bin/env fish
# Quick start script for Smart Offer Finder

echo "🚀 Smart Offer Finder - Start Script"
echo "===================================="
echo ""

# Check if virtual environment exists
if not test -d ".venv"
    echo "❌ Virtual environment not found!"
    echo "📖 Please run: python -m venv .venv"
    echo "   Then: source .venv/bin/activate.fish"
    exit 1
end

# Activate virtual environment
source .venv/bin/activate.fish

echo "✅ Virtual environment activated"
echo ""

# Check if dependencies are installed
echo "📦 Checking dependencies..."
if not python -c "import fastapi" 2>/dev/null
    echo "⚠️  Installing missing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
end

echo ""
echo "Starting servers..."
echo ""

# Start backend in background
echo "🔧 Starting FastAPI backend on http://localhost:8000..."
python main.py &
BACKEND_PID=$!

# Give backend time to start
sleep 2

# Start frontend
echo "🎨 Starting React frontend on http://localhost:3000..."
echo ""
echo "Frontend will open automatically. If not, visit: http://localhost:3000"
echo ""
echo "To stop: Press Ctrl+C"
echo ""

# Change to frontend directory and start
cd frontend
npm install --silent 2>/dev/null
npm start

# Cleanup when exiting
trap "kill $BACKEND_PID" EXIT
