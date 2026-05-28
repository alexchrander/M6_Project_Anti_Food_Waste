"""
Migrate legacy ChromaDB index_metadata.pickle files from plain dict format
to PersistentData class instances required by ChromaDB >= 0.5.x.

Background: older ChromaDB pickled plain dicts; newer versions expect
PersistentData objects with attribute access. This causes:
  AttributeError: 'dict' object has no attribute 'dimensionality'

Run once from the project root:
    python migrate_chroma_pickles.py

Safe to re-run: skips files that are already in the correct format.
"""

# ---------------------------------------------------------------------------
# Stub hnswlib BEFORE any chromadb imports so the module loads without error.
# We only need the class/attr names that chromadb references at import time.
# ---------------------------------------------------------------------------
import sys
from types import ModuleType

_hnswlib_stub = ModuleType("hnswlib")

class _FakeIndex:
    file_handle_count = 0

_hnswlib_stub.Index = _FakeIndex  # type: ignore[attr-defined]
sys.modules.setdefault("hnswlib", _hnswlib_stub)

# ---------------------------------------------------------------------------
# Now chromadb's local_persistent_hnsw can be imported cleanly.
# ---------------------------------------------------------------------------
import os
import pickle
import shutil
import sqlite3
import tempfile
from pathlib import Path

from chromadb.segment.impl.vector.local_persistent_hnsw import PersistentData

CHROMA_DB = Path(__file__).parent / "data" / "chroma_db"
SQLITE_DB = CHROMA_DB / "chroma.sqlite3"
DEFAULT_DIM = 3072  # gemini-embedding-001 output dimension


def get_dim_from_sqlite(segment_id: str) -> int:
    """Look up the collection's declared dimension for a given HNSW segment."""
    conn = sqlite3.connect(str(SQLITE_DB))
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.dimension
        FROM segments s
        JOIN collections c ON s.collection = c.id
        WHERE s.id = ?
        """,
        (segment_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] else DEFAULT_DIM


def is_already_migrated(data: object) -> bool:
    """Return True if the pickle already holds a proper PersistentData instance."""
    return isinstance(data, PersistentData)


def migrate_pickle(path: Path, segment_id: str) -> None:
    with open(path, "rb") as f:
        data = pickle.load(f)

    if is_already_migrated(data):
        dim = getattr(data, "dimensionality", None)
        if dim is not None:
            print(f"  {segment_id}: already PersistentData (dim={dim}) — skipping")
            return
        # dimensionality was stored as None; patch it in place.
        print(f"  {segment_id}: PersistentData with dim=None — patching")
        data.dimensionality = get_dim_from_sqlite(segment_id)
    elif isinstance(data, dict):
        dim = get_dim_from_sqlite(segment_id)
        print(
            f"  {segment_id}: dict → PersistentData "
            f"(dim={dim}, elements={data.get('total_elements_added')})"
        )
        data = PersistentData(
            dimensionality=dim,
            total_elements_added=data.get("total_elements_added", 0),
            id_to_label=data.get("id_to_label", {}),
            label_to_id=data.get("label_to_id", {}),
            id_to_seq_id=data.get("id_to_seq_id", {}),
        )
        # Keep None to match the original dict format; the Rust backend expects
        # an Optional value here and rejects negative integers (u64 field).
        data.max_seq_id = None
    else:
        print(f"  {segment_id}: unexpected type {type(data)} — skipping")
        return

    # Write atomically: temp file then rename, so a crash can't corrupt the DB.
    backup = path.with_suffix(".pickle.bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    tmp = path.with_suffix(".pickle.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

    print(f"    saved (backup: {backup.name})")


def main() -> None:
    print(f"ChromaDB path: {CHROMA_DB}")
    for uuid_dir in sorted(CHROMA_DB.iterdir()):
        if not uuid_dir.is_dir():
            continue
        meta_file = uuid_dir / "index_metadata.pickle"
        if not meta_file.exists():
            continue
        migrate_pickle(meta_file, uuid_dir.name)
    print("Done.")


if __name__ == "__main__":
    main()
