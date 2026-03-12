"""CLI entry point: sglang-tp-viz generate <model_id_or_path> [--output FILE]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config_loader import load_hf_config, load_local_config, detect_family
from .tp_analyzer import analyze


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sglang-tp-viz",
        description="Generate model architecture + TP layout JSON",
    )
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate architecture JSON")
    gen.add_argument(
        "model",
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B) "
        "or path to a local config.json",
    )
    gen.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: stdout)",
    )
    gen.add_argument(
        "--family",
        help="Override auto-detected model family (llama, deepseek_v2)",
    )

    args = parser.parse_args(argv)

    if args.command != "generate":
        parser.print_help()
        sys.exit(1)

    model = args.model
    local_path = Path(model)

    if local_path.exists():
        # Local config.json
        hf_config = load_local_config(local_path)
        model_id = local_path.stem
    else:
        # HuggingFace Hub
        hf_config = load_hf_config(model)
        model_id = model

    family = args.family or detect_family(hf_config)
    result = analyze(model_id=model_id, hf_config=hf_config, family=family)
    json_str = result.model_dump_json(indent=2)

    if args.output:
        Path(args.output).write_text(json_str)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
