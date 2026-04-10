from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


BASE = "https://huggingface.co/shadowlilac/gemma-4-e4b-mtp-extraction-effort/resolve/main"
DEFAULT_ARTIFACTS = {
    "model.toml": f"{BASE}/model.toml",
    "LlmMetadataProto.pbtext": f"{BASE}/LlmMetadataProto.pbtext",
    "extracted/mtp_graph_json_aiedge_model_explorer_extracted.json": (
        f"{BASE}/extracted/mtp_graph_json_aiedge_model_explorer_extracted.json"
    ),
}
OPTIONAL_ARTIFACTS = {
    "Section11_TFLiteModel_tf_lite_mtp_drafter.tflite": (
        f"{BASE}/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite"
    ),
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    destination.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()[:16]
    print(f"saved {destination} ({len(payload)} bytes, sha256={digest})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/hf",
        help="Directory to store downloaded artifacts.",
    )
    parser.add_argument(
        "--include-tflite",
        action="store_true",
        help="Also download the mtp_drafter.tflite artifact (~45 MB).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for relative_path, url in DEFAULT_ARTIFACTS.items():
        download(url, output_dir / relative_path)
    if args.include_tflite:
        for relative_path, url in OPTIONAL_ARTIFACTS.items():
            download(url, output_dir / relative_path)


if __name__ == "__main__":
    main()
