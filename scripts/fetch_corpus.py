"""Download the PEP corpus from a pinned commit and record the source of each file.

Run this only to change the corpus. The fetched files are committed, so the rest of
the pipeline never needs the network to read them.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

from cite_rag.corpus import detect_license, parse_headers

PEPS_COMMIT = "28581cca58bda3ad476e8a210be6df4733e88c97"
RAW_URL = (
    "https://raw.githubusercontent.com/python/peps/{commit}/peps/pep-{number:04d}.rst"
)

PEP_NUMBERS = [
    8, 20, 257, 318, 343, 380, 420, 440, 484, 492, 498, 517, 518, 526, 544, 557, 561,
    563, 570, 572, 585, 589, 591, 593, 604, 612, 621, 634, 635, 636, 654, 668, 673, 695,
    701, 703, 723, 3107, 3119, 3129, 3135,
]  # fmt: skip

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


def fetch(number: int) -> tuple[str, str]:
    url = RAW_URL.format(commit=PEPS_COMMIT, number=number)
    with urllib.request.urlopen(url, timeout=30) as response:
        return url, response.read().decode("utf-8")


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    sources = []
    for number in PEP_NUMBERS:
        url, text = fetch(number)
        license_name = detect_license(text)
        if license_name is None:
            print(f"skipped PEP {number}: no public-domain or CC0 statement")
            continue
        file_name = f"pep-{number:04d}.rst"
        (CORPUS_DIR / file_name).write_bytes(text.encode("utf-8"))
        headers = parse_headers(text)
        sources.append(
            {
                "file": file_name,
                "pep": number,
                "title": headers["Title"],
                "status": headers["Status"],
                "license": license_name,
                "source_url": url,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    manifest = CORPUS_DIR / "sources.json"
    manifest.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(sources)} PEPs to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
