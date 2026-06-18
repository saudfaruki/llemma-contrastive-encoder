"""Resume-safe, idempotent Llemma prefetch for the 4090 bootstrap.

Bakes in the download lessons learned on the dev Mac:
- HF_HUB_DOWNLOAD_TIMEOUT so a stalled socket times out (instead of deadlocking).
- A retry loop so transient connection drops self-heal — snapshot_download resumes
  from the on-disk partial each attempt. Already-cached weights make this a no-op.
Llemma is public, so no token is required; HF_TOKEN is used if present (rate limits).
"""
import os
import time

MODEL_ID = "EleutherAI/llemma_7b"


def fetch(model_id: str = MODEL_ID, max_attempts: int = 300, sleep_s: int = 5) -> str:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from huggingface_hub import snapshot_download
    for attempt in range(1, max_attempts + 1):
        try:
            path = snapshot_download(model_id)
            print(f"SNAPSHOT_OK {path}", flush=True)
            return path
        except Exception as e:  # noqa: BLE001 — any drop is retryable; resume from partial
            print(f"[attempt {attempt}] download dropped: {e}; resuming in {sleep_s}s",
                  flush=True)
            time.sleep(sleep_s)
    raise SystemExit(f"FATAL: {model_id} did not finish after {max_attempts} attempts")


if __name__ == "__main__":
    fetch()
