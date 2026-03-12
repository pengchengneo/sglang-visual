"""Core orchestrator: config → template → ModelArchitecture JSON."""

from __future__ import annotations

from .config_loader import detect_family, parse_model_config
from .families.base import ModelFamilyTemplate
from .families.deepseek_v2 import DeepSeekV2Template
from .families.llama import LlamaTemplate
from .schema import CommSummary, ModelArchitecture

_TEMPLATE_REGISTRY: dict[str, type[ModelFamilyTemplate]] = {
    "llama": LlamaTemplate,
    "deepseek_v2": DeepSeekV2Template,
}


def analyze(
    model_id: str,
    hf_config: dict,
    family: str | None = None,
) -> ModelArchitecture:
    """Generate the full ModelArchitecture from an HF config dict.

    Args:
        model_id: HuggingFace model identifier (e.g. "meta-llama/Llama-3.1-8B")
        hf_config: Raw HF config.json dict
        family: Optional override; auto-detected if None
    """
    if family is None:
        family = detect_family(hf_config)

    template_cls = _TEMPLATE_REGISTRY.get(family)
    if template_cls is None:
        raise ValueError(
            f"No template for family '{family}'. "
            f"Available: {list(_TEMPLATE_REGISTRY.keys())}"
        )

    config = parse_model_config(hf_config, family)
    template = template_cls(config)

    embedding = template.get_embedding()
    lm_head = template.get_lm_head()
    layers = template.get_all_layers()

    # Communication summary
    if layers:
        ops_per_layer = template.count_comm_ops_per_layer(layers[0])
    else:
        ops_per_layer = 0
    # For MoE models, dense and MoE layers may differ; count worst case
    total_ops = sum(template.count_comm_ops_per_layer(l) for l in layers)
    # Add embedding all-reduce
    if embedding.comm_after:
        total_ops += 1

    return ModelArchitecture(
        model_id=model_id,
        model_family=family,
        tp_size=1,
        config=config,
        embedding=embedding,
        lm_head=lm_head,
        layers=layers,
        communication_summary=CommSummary(
            ops_per_layer=ops_per_layer,
            total_ops=total_ops,
        ),
    )
