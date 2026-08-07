import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class SemanticStore:
    """
    Semantic Memory Store for keeping track of persistent facts and knowledge.
    KEY: This store is NEVER written to directly by the router. Only the consolidation layer writes here.
    Backed by SQLite database memory.db.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT NOT NULL,
                    category TEXT NOT NULL,
                    entity_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER DEFAULT 1,
                    expires_at DATETIME,
                    source_episode_ids TEXT,
                    status TEXT DEFAULT 'active',
                    superseded_by INTEGER,
                    conflict_resolution_note TEXT
                )
            ''')
            conn.commit()

    def _row_to_dict(self, row) -> Dict[str, Any]:
        return {
            "id": row[0],
            "fact": row[1],
            "category": row[2],
            "entity_id": row[3],
            "created_at": row[4],
            "updated_at": row[5],
            "version": row[6],
            "expires_at": row[7],
            "source_episode_ids": json.loads(row[8]) if row[8] else [],
            "status": row[9],
            "superseded_by": row[10],
            "conflict_resolution_note": row[11]
        }

    def add_fact(self, fact: str, category: str, entity_id: Optional[str] = None, 
                 source_episode_ids: List[int] = None, expires_in_days: Optional[int] = None) -> int:
        """Insert new fact, returns fact_id."""
        source_ids_str = json.dumps(source_episode_ids or [])
        expires_at = None
        if expires_in_days:
            expires_at = (datetime.utcnow() + timedelta(days=expires_in_days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO semantic_memory 
                (fact, category, entity_id, expires_at, source_episode_ids)
                VALUES (?, ?, ?, ?, ?)
            ''', (fact, category, entity_id, expires_at, source_ids_str))
            conn.commit()
            return cursor.lastrowid

    def update_fact(self, old_fact_id: int, new_fact: str, source_episode_ids: List[int], 
                    conflict_note: str) -> int:
        """
        Supersedes old fact, creates new versioned one.
        Old fact status becomes 'superseded' with pointer to new.
        NEW fact gets version = old_version + 1.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get old fact details
            cursor.execute('SELECT version, category, entity_id, expires_at FROM semantic_memory WHERE id = ?', (old_fact_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Fact with ID {old_fact_id} not found")
                
            old_version, category, entity_id, expires_at = row
            new_version = old_version + 1
            source_ids_str = json.dumps(source_episode_ids or [])
            
            # Insert new fact
            cursor.execute('''
                INSERT INTO semantic_memory 
                (fact, category, entity_id, version, expires_at, source_episode_ids, conflict_resolution_note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (new_fact, category, entity_id, new_version, expires_at, source_ids_str, conflict_note))
            
            new_fact_id = cursor.lastrowid
            
            # Mark old fact as superseded
            cursor.execute('''
                UPDATE semantic_memory
                SET status = 'superseded', superseded_by = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (new_fact_id, old_fact_id))
            
            conn.commit()
            return new_fact_id

    def query_facts(self, entity_id: str, category: Optional[str] = None, include_expired: bool = False) -> List[Dict[str, Any]]:
        """Get active facts, optionally including expired."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM semantic_memory WHERE entity_id = ? AND status != 'superseded'"
            params = [entity_id]
            
            if category:
                query += " AND category = ?"
                params.append(category)
                
            if not include_expired:
                query += " AND status = 'active'"
                
            cursor.execute(query, tuple(params))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_fact_history(self, entity_id: str, category: str) -> List[Dict[str, Any]]:
        """Returns ALL versions of facts for an entity, including superseded ones (showing version chain)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM semantic_memory
                WHERE entity_id = ? AND category = ?
                ORDER BY version ASC, created_at ASC
            ''', (entity_id, category))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def expire_stale_facts(self) -> None:
        """Marks facts past their expires_at as 'expired'."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            current_time = datetime.utcnow().isoformat()
            cursor.execute('''
                UPDATE semantic_memory
                SET status = 'expired', updated_at = CURRENT_TIMESTAMP
                WHERE status = 'active' AND expires_at IS NOT NULL AND expires_at < ?
            ''', (current_time,))
            conn.commit()

    def search_facts(self, query_text: str) -> List[Dict[str, Any]]:
        """Text search across active facts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            search_pattern = f"%{query_text}%"
            cursor.execute('''
                SELECT * FROM semantic_memory
                WHERE status = 'active' AND fact LIKE ?
            ''', (search_pattern,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]
