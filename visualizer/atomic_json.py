"""Small atomic JSON writers used by visualizer post-processing scripts."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _atomic_json_write(path, write_json):
    """Write beside ``path`` and atomically replace it after a complete flush."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            write_json(fh)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    finally:
        # ``os.replace`` removes the temporary name.  On serialization or I/O
        # failure, remove the incomplete sibling and leave the target untouched.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def atomic_dump_json(path, payload):
    _atomic_json_write(path, lambda fh: json.dump(payload, fh))


def atomic_dump_data_js(path, prefix, payload):
    def write(fh):
        fh.write(prefix)
        json.dump(payload, fh)
        fh.write(";\n")

    _atomic_json_write(path, write)
