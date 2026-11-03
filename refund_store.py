
import sqlite3
import json
import uuid

from pathlib import Path
from datetime import datetime, timezone


PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "data" / "shopease_refunds.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS refund_requests (
                request_id TEXT PRIMARY KEY,
                thread_id TEXT,
                order_id TEXT NOT NULL,
                customer_unique_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                refund_amount REAL NOT NULL,
                currency TEXT NOT NULL,
                eligibility_status TEXT NOT NULL,
                eligibility_reasons TEXT NOT NULL,
                policy_evidence TEXT NOT NULL,
                approval_status TEXT NOT NULL,
                execution_status TEXT NOT NULL,
                reviewer_reason TEXT,
                reviewer_id TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT
            )
        """)

        # Migrate existing databases without deleting existing records.
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(refund_requests)"
            ).fetchall()
        }

        if "thread_id" not in columns:
            connection.execute(
                "ALTER TABLE refund_requests ADD COLUMN thread_id TEXT"
            )


def save_refund_request(refund_request: dict) -> str:
    initialize_database()

    request_id = str(uuid.uuid4())
    thread_id = f"refund-{request_id}"
    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        connection.execute("""
            INSERT INTO refund_requests (
                request_id, thread_id, order_id, customer_unique_id,
                reason, refund_amount, currency,
                eligibility_status, eligibility_reasons,
                policy_evidence, approval_status,
                execution_status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id,
            thread_id,
            refund_request["order_id"],
            refund_request["customer_unique_id"],
            refund_request["reason"],
            refund_request["refund_amount"],
            refund_request["currency"],
            refund_request["eligibility_status"],
            json.dumps(refund_request["eligibility_reasons"]),
            refund_request["policy_evidence"],
            "PENDING",
            "NOT_EXECUTED",
            created_at
        ))

    return request_id


def get_pending_refunds():
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT * FROM refund_requests
            WHERE approval_status = 'PENDING'
            ORDER BY created_at DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_refund_by_id(request_id: str):
    initialize_database()

    with get_connection() as connection:
        row = connection.execute("""
            SELECT * FROM refund_requests
            WHERE request_id = ?
        """, (request_id,)).fetchone()

    return dict(row) if row else None


def update_refund_decision(
    request_id: str,
    decision: str,
    reviewer_id: str,
    reviewer_reason: str
):
    """
    Temporary legacy database update.
    We will replace dashboard usage with LangGraph workflow resumption.
    """

    decision = decision.upper()

    if decision not in ["APPROVED", "REJECTED"]:
        raise ValueError("Decision must be APPROVED or REJECTED.")

    execution_status = (
        "SIMULATED_SUCCESS" if decision == "APPROVED"
        else "NOT_EXECUTED"
    )

    reviewed_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute("""
            UPDATE refund_requests
            SET approval_status = ?,
                execution_status = ?,
                reviewer_id = ?,
                reviewer_reason = ?,
                reviewed_at = ?
            WHERE request_id = ?
              AND approval_status = 'PENDING'
              AND eligibility_status = 'ELIGIBLE'
        """, (
            decision,
            execution_status,
            reviewer_id,
            reviewer_reason,
            reviewed_at,
            request_id
        ))

        if cursor.rowcount == 0:
            raise ValueError(
                "Request not found, not eligible, or already reviewed."
            )
def set_refund_thread_id(request_id: str, thread_id: str):
    initialize_database()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE refund_requests
            SET thread_id = ?
            WHERE request_id = ?
            """,
            (thread_id, request_id)
        )

        if cursor.rowcount != 1:
            raise ValueError(f"Could not link thread to refund {request_id}")

def get_all_refunds():
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM refund_requests
            ORDER BY created_at DESC
        """).fetchall()

    return [dict(row) for row in rows]