from __future__ import annotations

from pathlib import Path

import torch

from gemma_mtp import GemmaMtpDrafter, infer_config_from_graph_export


def main() -> None:
    graph_path = Path("data/hf/extracted/mtp_graph_json_aiedge_model_explorer_extracted.json")
    if not graph_path.exists():
        raise SystemExit(
            "Missing graph export. Run `PYTHONPATH=src python scripts/fetch_artifacts.py` first."
        )

    config = infer_config_from_graph_export(graph_path)
    model = GemmaMtpDrafter(config)

    activations = torch.randn(1, 1, config.input_activation_dim)
    mask = torch.zeros(1, 1, 1, 32003, dtype=torch.bool)
    output = model(activations, mask=mask, base_kv_cache={})

    print("logits", tuple(output.logits.shape))
    print("projected_activations", tuple(output.projected_activations.shape))
    print("hidden_states", tuple(output.hidden_states.shape))


if __name__ == "__main__":
    main()
