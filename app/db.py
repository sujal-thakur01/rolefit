from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create the extension, table, generated tsvector column and indexes.

    Written as raw DDL rather than SQLAlchemy metadata because the tsvector
    column is generated and the vector index needs explicit operator classes.
    """
    dim = settings.embedding_dim
    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS jobs (
        id              bigserial PRIMARY KEY,
        source          text        NOT NULL,
        source_id       text        NOT NULL,
        posted_at       timestamptz,
        raw_text        text        NOT NULL,

        title           text,
        company         text,
        location        text,
        work_mode       text,
        seniority       text,
        stack           text[]      NOT NULL DEFAULT '{{}}',
        comp            text,
        visa            text,
        summary         text,
        intern_friendly boolean,

        extraction_ok   boolean     NOT NULL DEFAULT false,
        embed_text      text,
        embedding       vector({dim}),
        created_at      timestamptz NOT NULL DEFAULT now(),

        CONSTRAINT jobs_source_uniq UNIQUE (source, source_id)
    );

    ALTER TABLE jobs
        ADD COLUMN IF NOT EXISTS search_doc tsvector
        GENERATED ALWAYS AS (
            setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
            setweight(to_tsvector('english', coalesce(company, '')), 'A') ||
            setweight(to_tsvector('english', array_to_string(stack, ' ')), 'B') ||
            setweight(to_tsvector('english', coalesce(summary, '')), 'C') ||
            setweight(to_tsvector('english', coalesce(location, '')), 'C')
        ) STORED;

    CREATE INDEX IF NOT EXISTS jobs_search_doc_idx ON jobs USING gin (search_doc);
    CREATE INDEX IF NOT EXISTS jobs_embedding_idx
        ON jobs USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS jobs_posted_at_idx ON jobs (posted_at DESC);
    """
    with engine.begin() as conn:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            conn.execute(text(stmt))
