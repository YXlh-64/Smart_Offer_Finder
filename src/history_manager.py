"""
Chat History Manager - SQLite persistence for conversations.
Handles CRUD operations for sessions and messages.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class Message:
    """Represents a chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    sources: Optional[list[str]] = None


@dataclass
class Session:
    """Represents a chat session"""
    id: str
    title: str
    created_at: str
    updated_at: str


class HistoryManager:
    """Manages persistent chat history using SQLite"""
    
    def __init__(self, db_path: str = "data/chat_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id)
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, title: str = "Nouvelle conversation") -> Session:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now)
        )
        conn.commit()
        conn.close()
        
        return Session(id=session_id, title=title, created_at=now, updated_at=now)
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Session(
                id=row['id'],
                title=row['title'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
        return None
    
    def get_all_sessions(self) -> list[Session]:
        """Get all sessions ordered by most recent"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Session(
                id=row['id'],
                title=row['title'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            for row in rows
        ]
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update a session's title"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, datetime.now().isoformat(), session_id)
        )
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Delete messages first (foreign key)
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    def save_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        sources: Optional[list[str]] = None
    ) -> Message:
        """Save a message to a session"""
        now = datetime.now().isoformat()
        sources_str = ",".join(sources) if sources else None
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Insert message
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, sources, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, sources_str, now)
        )
        
        # Update session's updated_at
        cursor.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id)
        )
        
        conn.commit()
        conn.close()
        
        return Message(role=role, content=content, timestamp=now, sources=sources)
    
    def save_turn(
        self, 
        session_id: str, 
        user_message: str, 
        ai_message: str,
        sources: Optional[list[str]] = None
    ):
        """Save a complete turn (user + AI messages)"""
        self.save_message(session_id, "user", user_message)
        self.save_message(session_id, "assistant", ai_message, sources)
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> list[Message]:
        """Get all messages for a session"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if limit:
            cursor.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
                (session_id, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            sources = row['sources'].split(",") if row['sources'] else None
            messages.append(Message(
                role=row['role'],
                content=row['content'],
                timestamp=row['timestamp'],
                sources=sources
            ))
        
        return messages
    
    def get_recent_history(self, session_id: str, turns: int = 5) -> list[tuple[str, str]]:
        """
        Get recent chat history as (human, ai) tuples for LangChain context.
        Returns the last N turns (pairs of user-assistant messages).
        """
        messages = self.get_messages(session_id)
        
        # Group into turns
        history = []
        i = 0
        while i < len(messages) - 1:
            if messages[i].role == "user" and messages[i + 1].role == "assistant":
                history.append((messages[i].content, messages[i + 1].content))
                i += 2
            else:
                i += 1
        
        # Return only the last N turns
        return history[-turns:] if turns else history
    
    def generate_title_from_message(self, message: str, max_length: int = 40) -> str:
        """Generate a session title from the first user message"""
        # Clean and truncate the message
        title = message.strip()
        title = " ".join(title.split())  # Normalize whitespace
        
        if len(title) > max_length:
            title = title[:max_length - 3] + "..."
        
        return title if title else "Nouvelle conversation"


# Global instance
_history_manager: Optional[HistoryManager] = None


def get_history_manager() -> HistoryManager:
    """Get the global history manager instance"""
    global _history_manager
    if _history_manager is None:
        _history_manager = HistoryManager()
    return _history_manager
