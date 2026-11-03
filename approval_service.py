
from pathlib import Path
from contextlib import contextmanager

from langgraph.checkpoint.sqlite import SqliteSaver
from refund_approval import build_refund_approval_graph


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DB = PROJECT_DIR / "data" / "refund_checkpoints.sqlite"


@contextmanager
def get_approval_graph():
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        graph = build_refund_approval_graph(checkpointer)
        yield graph
