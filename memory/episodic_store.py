import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class EpisodicStore:
    """
    Episodic Memory Store for keeping track of discrete events and occurrences.
    Backed by SQLite database memory.db.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    node_id TEXT,
                    customer_id TEXT,
                    importance_score REAL DEFAULT 0.5,
                    metadata TEXT,
                    consolidated BOOLEAN DEFAULT FALSE
                )
            ''')
            conn.commit()

    def store(self, session_id: str, event_type: str, content: str, 
              node_id: Optional[str] = None, customer_id: Optional[str] = None, 
              importance: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Insert a new episode into the episodic memory store."""
        meta_str = json.dumps(metadata) if metadata else "{}"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO episodic_memory 
                (session_id, event_type, content, node_id, customer_id, importance_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, event_type, content, node_id, customer_id, importance, meta_str))
            conn.commit()
            return cursor.lastrowid

    def _row_to_dict(self, row) -> Dict[str, Any]:
        return {
            "id": row[0],
            "timestamp": row[1],
            "session_id": row[2],
            "event_type": row[3],
            "content": row[4],
            "node_id": row[5],
            "customer_id": row[6],
            "importance_score": row[7],
            "metadata": json.loads(row[8]) if row[8] else {},
            "consolidated": bool(row[9])
        }

    def query_by_entity(self, entity_type: str, entity_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get episodes for a node or customer."""
        column = "node_id" if entity_type == "node" else "customer_id"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT id, timestamp, session_id, event_type, content, node_id, customer_id, 
                       importance_score, metadata, consolidated
                FROM episodic_memory
                WHERE {column} = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (entity_id, limit))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def query_by_type(self, event_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get episodes by type."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, session_id, event_type, content, node_id, customer_id, 
                       importance_score, metadata, consolidated
                FROM episodic_memory
                WHERE event_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (event_type, limit))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def query_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Most recent episodes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, session_id, event_type, content, node_id, customer_id, 
                       importance_score, metadata, consolidated
                FROM episodic_memory
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def query_unconsolidated(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Episodes not yet processed by consolidation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, session_id, event_type, content, node_id, customer_id, 
                       importance_score, metadata, consolidated
                FROM episodic_memory
                WHERE consolidated = 0
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (limit,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def mark_consolidated(self, episode_ids: List[int]) -> None:
        """Mark episodes as consolidated."""
        if not episode_ids:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(episode_ids))
            cursor.execute(f'''
                UPDATE episodic_memory
                SET consolidated = 1
                WHERE id IN ({placeholders})
            ''', tuple(episode_ids))
            conn.commit()

    def search_text(self, query_text: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Basic text search in content."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            search_pattern = f"%{query_text}%"
            cursor.execute('''
                SELECT id, timestamp, session_id, event_type, content, node_id, customer_id, 
                       importance_score, metadata, consolidated
                FROM episodic_memory
                WHERE content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (search_pattern, limit))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_all(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all episodes up to limit."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, session_id, event_type, content, node_id, customer_id, 
                       importance_score, metadata, consolidated
                FROM episodic_memory
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]
