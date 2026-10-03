"""Sections 6 and 7 — SQL agent with a read-only gate, plus a sandboxed chart.

The guide's create_sql_agent can run arbitrary SQL. This script keeps that
pipeline (inspect schema, write SQL, execute, answer) but checks the statement
before it reaches SQLite. The chart is drawn by this process from query rows.
The model never receives a Python exec tool.
"""

import re
import sqlite3
from pathlib import Path

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from llm_factory import make_llm, visible_text

DB_PATH = Path(__file__).resolve().parent / "data" / "music.sqlite"
QUESTION = "How many albums are in the database?"
EXPECTED_ALBUMS = 8
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|pragma|create|replace|vacuum)\b",
    re.IGNORECASE,
)


def build_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    connection = sqlite3.connect(DB_PATH)
    connection.executescript(
        """
        CREATE TABLE artists (
            ArtistId INTEGER PRIMARY KEY,
            Name TEXT NOT NULL
        );
        CREATE TABLE albums (
            AlbumId INTEGER PRIMARY KEY,
            Title TEXT NOT NULL,
            ArtistId INTEGER NOT NULL,
            FOREIGN KEY (ArtistId) REFERENCES artists(ArtistId)
        );
        INSERT INTO artists (ArtistId, Name) VALUES
            (1, 'The Beatles'),
            (2, 'Radiohead'),
            (3, 'Adele');
        INSERT INTO albums (AlbumId, Title, ArtistId) VALUES
            (1, 'Abbey Road', 1),
            (2, 'Revolver', 1),
            (3, 'Let It Be', 1),
            (4, 'OK Computer', 2),
            (5, 'In Rainbows', 2),
            (6, '19', 3),
            (7, '21', 3),
            (8, '25', 3);
        """
    )
    connection.commit()
    connection.close()


def validate_select(query: str) -> str | None:
    statement = query.strip().rstrip(";")
    if not statement.lower().startswith("select"):
        return "only a single SELECT is allowed"
    if ";" in statement:
        return "only one statement is allowed"
    if FORBIDDEN.search(statement):
        return "write or admin SQL is blocked"
    return None


def query_rows(query: str) -> list[tuple]:
    reason = validate_select(query)
    if reason:
        raise ValueError(reason)
    connection = sqlite3.connect(DB_PATH)
    try:
        return connection.execute(statement_text(query)).fetchall()
    finally:
        connection.close()


def statement_text(query: str) -> str:
    return query.strip().rstrip(";")


@tool
def describe_schema() -> str:
    """Return the SQLite table definitions for the music database."""
    print("[middleware] before describe_schema")
    connection = sqlite3.connect(DB_PATH)
    try:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        connection.close()
    schema = "\n".join(row[0] for row in rows)
    print("[middleware] after describe_schema")
    return schema


@tool
def run_select(query: str) -> str:
    """Run one read-only SELECT against the music database."""
    print(f"[middleware] before run_select: {query}")
    reason = validate_select(query)
    if reason:
        print(f"[middleware] reject run_select - {reason}")
        return f"rejected: {reason}"
    rows = query_rows(query)
    print(f"[middleware] after run_select: {rows}")
    return str(rows)


TOOLS = {"describe_schema": describe_schema, "run_select": run_select}


def ascii_chart() -> None:
    rows = query_rows(
        """
        SELECT artists.Name, COUNT(albums.AlbumId) AS albums
        FROM artists
        LEFT JOIN albums ON albums.ArtistId = artists.ArtistId
        GROUP BY artists.Name
        ORDER BY artists.Name
        """
    )
    print("\nsandboxed chart: albums per artist")
    for name, count in rows:
        print(f"  {name:<14} {'#' * int(count)} {count}")


def demo_blocked_sql() -> None:
    print("\nunsafe SQL check")
    observation = run_select.invoke({"query": "DROP TABLE albums"})
    print(f"observation: {observation}")
    if not str(observation).startswith("rejected:"):
        raise SystemExit("DROP TABLE was not blocked")
    remaining = query_rows("SELECT COUNT(*) FROM albums")[0][0]
    if remaining != EXPECTED_ALBUMS:
        raise SystemExit("albums table was modified")
    print(f"albums still present: {remaining}")


def main() -> None:
    build_database()
    llm = make_llm(num_predict=768).bind_tools([describe_schema, run_select])
    messages = [
        SystemMessage(
            content=(
                "You query a SQLite music database. Do not explain. "
                "Your first action must be the describe_schema tool. "
                "After the schema, call run_select with one SELECT. "
                "After the rows come back, answer in one sentence."
            )
        ),
        HumanMessage(content=QUESTION),
    ]
    print(f"question: {QUESTION}")

    answer = ""
    schema_seen = False
    rows_seen = False
    nudges = 0
    for _ in range(6):
        ai = llm.invoke(messages)
        if not ai.tool_calls:
            if not rows_seen and nudges < 2:
                nudges += 1
                hint = (
                    "Call run_select with one SELECT that counts albums."
                    if schema_seen
                    else "Call describe_schema now. Do not explain."
                )
                print(f"nudge: {hint}")
                messages.append(HumanMessage(content=hint))
                continue
            answer = visible_text(ai.content)
            break
        messages.append(ai)
        for call in ai.tool_calls:
            print(f"action: {call['name']}({call['args']})")
            observation = TOOLS[call["name"]].invoke(call["args"])
            print(f"observation: {observation}")
            if call["name"] == "describe_schema":
                schema_seen = True
            if call["name"] == "run_select" and not str(observation).startswith("rejected"):
                rows_seen = True
            messages.append(
                ToolMessage(content=str(observation), tool_call_id=call["id"])
            )
    else:
        raise SystemExit("agent did not finish within 6 steps")

    print(f"answer: {answer}")
    print(f"expected albums: {EXPECTED_ALBUMS}")
    if str(EXPECTED_ALBUMS) not in answer:
        raise SystemExit("final answer did not contain the album count")
    print("pass@1: pass")
    ascii_chart()
    demo_blocked_sql()


if __name__ == "__main__":
    main()
