#!/usr/bin/env python3
"""Create a dev-as-test view for threshold calibration without changing labels."""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    with (source / "dev.pkl").open("rb") as f:
        payload = pickle.load(f, encoding="latin-1")
    if "dev" not in payload:
        raise KeyError(f"{source / 'dev.pkl'} does not contain the 'dev' split")
    test_payload = {
        key: value for key, value in payload.items()
        if key != "dev"
    }
    test_payload["test"] = payload["dev"]
    with (output / "test.pkl").open("wb") as f:
        pickle.dump(test_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    for filename in (
        "train.pkl",
        "dev.pkl",
        "annotation.csv",
        "metadata.json",
        "vocab_templates.json",
        "state_aware_tokens.json",
    ):
        src = source / filename
        if src.exists():
            shutil.copy2(src, output / filename)

    print(f"[OK] Created dev-as-test dataset at {output}")
    print(f"[OK] dev sequences exposed as test: {len(payload['dev'])}")


if __name__ == "__main__":
    main()
