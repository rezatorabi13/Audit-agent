"""Generate the synthetic engagement corpus and build the vector index.

Run this once before the other examples:

    python examples/00_setup_corpus.py

Re-running is safe: the corpus is overwritten and the index is rebuilt from
scratch, so it is also the way to recover from a partial or stale ingest.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running a script puts examples/ on sys.path, not the repository root, so the
# package would not resolve without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_agent.config import get_settings  # noqa: E402
from audit_agent.data.synthetic import generate_corpus  # noqa: E402
from audit_agent.logging_config import configure_logging  # noqa: E402
from audit_agent.pipeline import ingest_corpus  # noqa: E402


def main() -> None:
    configure_logging()
    settings = get_settings()

    generate_corpus(settings.docs_dir, settings.tables_dir)
    print(f"Corpus written to {settings.docs_dir} and {settings.tables_dir}")

    chunk_count = ingest_corpus(settings, reset=True)
    print(f"Indexed {chunk_count} chunks into {settings.chroma_dir}")


if __name__ == "__main__":
    main()