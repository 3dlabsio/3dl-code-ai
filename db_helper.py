import os
from datetime import datetime
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from contextlib import contextmanager

class DatabaseHelper:
    def __init__(self):
        self.conn_string = os.getenv('DATABASE_URL')
        if not self.conn_string:
            raise ValueError("DATABASE_URL environment variable not set")
        self._init_db()

    def _init_db(self):
        """Initialize the database schema"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        dataset_path TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        title TEXT,
                        is_deleted BOOLEAN DEFAULT FALSE
                    );
                    
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        context_chunks JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_session_id 
                    ON chat_messages(session_id);
                """)
                conn.commit()

    @contextmanager
    def _get_conn(self):
        """Context manager for database connections"""
        conn = psycopg2.connect(self.conn_string)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def create_session(self, session_id: str, dataset_path: str, title: str = None):
        """Create a new chat session"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO chat_sessions 
                    (session_id, dataset_path, title) 
                    VALUES (%s, %s, %s)""",
                    (session_id, dataset_path, title or "New Chat")
                )

    def get_all_sessions(self):
        """Get all active chat sessions"""
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT session_id, dataset_path, title, created_at 
                    FROM chat_sessions 
                    WHERE NOT is_deleted 
                    ORDER BY created_at DESC"""
                )
                return cur.fetchall()

    def delete_session(self, session_id: str):
        """Soft delete a chat session"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chat_sessions SET is_deleted = TRUE WHERE session_id = %s",
                    (session_id,)
                )

    def update_session_title(self, session_id: str, title: str):
        """Update the title of a chat session"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chat_sessions SET title = %s WHERE session_id = %s",
                    (title, session_id)
                )

    def update_session_dataset(self, session_id: str, dataset_path: str):
        """Update the dataset path for a session"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chat_sessions SET dataset_path = %s WHERE session_id = %s",
                    (dataset_path, session_id)
                )

    def get_session_info(self, session_id: str):
        """Get session information"""
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT dataset_path, title 
                    FROM chat_sessions 
                    WHERE session_id = %s""",
                    (session_id,)
                )
                return cur.fetchone()

    def add_message(self, session_id: str, role: str, content: str, context_chunks=None):
        """Add a new message to the chat history"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO chat_messages 
                    (session_id, role, content, context_chunks) 
                    VALUES (%s, %s, %s, %s)""",
                    (session_id, role, content, Json(context_chunks) if context_chunks else None)
                )

    def get_session_history(self, session_id: str, limit: int = 10):
        """Get recent chat history for a session"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT role, content, context_chunks 
                    FROM chat_messages 
                    WHERE session_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s""",
                    (session_id, limit)
                )
                return cur.fetchall()

    def get_context_history(self, session_id: str, limit: int = 3):
        """Get recent context chunks for a session"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT context_chunks 
                    FROM chat_messages 
                    WHERE session_id = %s 
                    AND context_chunks IS NOT NULL 
                    ORDER BY created_at DESC 
                    LIMIT %s""",
                    (session_id, limit)
                )
                return [row[0] for row in cur.fetchall() if row[0]]