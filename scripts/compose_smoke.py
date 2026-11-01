"""Smoke test for the running Docker Compose stack (db + api + ui).

Proves the containers are wired together without any provider key: the API answers,
its migrations ran against the `db` service (a collection is created and a document
is ingested), PostgreSQL full-text search returns the expected chunk, and the
Streamlit UI is up. Embeddings, reranking and generation need a provider and are
covered by the test suite instead.

It leaves a `compose-smoke-<hex>` collection behind, so run it against a throwaway
stack: CI tears its stack down with `--volumes`, and `make compose-smoke` uses a
separate Compose project that it removes afterwards.

Standard library only, so it runs on the host with any Python 3.10+:

    python3 scripts/compose_smoke.py [--api-url ...] [--ui-url ...]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# The query and the text it must rank first are tied to this sample document.
DOCUMENT = Path(__file__).resolve().parent.parent / "examples/corpus/returns-and-refunds.md"
QUERY = "Final Sale tag"
EXPECTED_TEXT = "Final Sale"


class SmokeError(Exception):
    pass


def request(
    method: str, url: str, body: bytes | None = None, headers: dict[str, str] | None = None
) -> Any:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            payload = response.read().decode()
    except urllib.error.HTTPError as error:
        raise SmokeError(f"{method} {url} -> {error.code}: {error.read().decode()}") from error
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload


def post_json(url: str, data: dict[str, Any]) -> Any:
    return request("POST", url, json.dumps(data).encode(), {"Content-Type": "application/json"})


def upload(url: str, path: Path) -> Any:
    boundary = uuid.uuid4().hex
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
            b"Content-Type: text/markdown\r\n\r\n",
            path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return request("POST", url, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def wait_until_up(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            request("GET", url)
            return
        except (SmokeError, OSError) as error:
            if time.monotonic() >= deadline:
                raise SmokeError(f"{url} not reachable after {timeout:.0f}s: {error}") from error
            time.sleep(2)


def run(api_url: str, ui_url: str, timeout: float) -> None:
    wait_until_up(f"{api_url}/health", timeout)
    print(f"ok  API is up at {api_url}")

    collection = post_json(
        f"{api_url}/collections", {"name": f"compose-smoke-{uuid.uuid4().hex[:8]}"}
    )
    chunk_set = collection["chunking_configs"][0]["key"]
    print(f"ok  collection {collection['id']} created (migrations applied)")

    uploaded = upload(f"{api_url}/collections/{collection['id']}/documents", DOCUMENT)
    print(f"ok  uploaded {DOCUMENT.name} as document {uploaded['id']}")

    result = post_json(
        f"{api_url}/collections/{collection['id']}/search",
        {"query": QUERY, "chunk_set": chunk_set, "method": "fulltext", "top_k": 3},
    )
    if not result["hits"] or EXPECTED_TEXT not in result["hits"][0]["text"]:
        raise SmokeError(f"full-text search for {QUERY!r} did not rank the expected chunk first")
    print(f"ok  full-text search on {chunk_set} found {EXPECTED_TEXT!r} at rank 1")

    wait_until_up(f"{ui_url}/_stcore/health", timeout)
    print(f"ok  UI is up at {ui_url}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--ui-url", default="http://localhost:8501")
    parser.add_argument(
        "--timeout", type=float, default=120, help="seconds to wait for each service"
    )
    args = parser.parse_args()
    try:
        run(args.api_url.rstrip("/"), args.ui_url.rstrip("/"), args.timeout)
    except SmokeError as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    except OSError as error:  # connection reset, timeout after the service came up
        print(f"FAIL network error: {error}", file=sys.stderr)
        return 1
    except (KeyError, IndexError, TypeError) as error:  # unexpected response shape
        print(f"FAIL unexpected API response: {error!r}", file=sys.stderr)
        return 1
    print("Compose smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
