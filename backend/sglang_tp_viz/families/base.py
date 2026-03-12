"""Abstract base class for model family TP templates."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import EmbeddingInfo, Layer, ModelConfig, PartitionStrategy, CommOp


class ModelFamilyTemplate(ABC):
    """Base class that each model family implements to describe its TP layout."""

    family_name: str = ""

    def __init__(self, config: ModelConfig):
        self.config = config

    def get_embedding(self) -> EmbeddingInfo:
        """Embedding layer: VocabParallelEmbedding, column-split by vocab."""
        c = self.config
        return EmbeddingInfo(
            full_shape=[c.vocab_size, c.hidden_size],
            tp_shape=[c.vocab_size, c.hidden_size],  # tp=1
            partition_strategy=PartitionStrategy.COLUMN,
            comm_after=CommOp.ALL_REDUCE,
        )

    def get_lm_head(self) -> EmbeddingInfo:
        """LM head: column-parallel, split output vocab dim."""
        c = self.config
        return EmbeddingInfo(
            full_shape=[c.vocab_size, c.hidden_size],
            tp_shape=[c.vocab_size, c.hidden_size],  # tp=1
            partition_strategy=PartitionStrategy.COLUMN,
            comm_after=None,
        )

    @abstractmethod
    def get_layer(self, layer_id: int) -> Layer:
        """Return the TP layout for a single decoder layer."""
        ...

    def get_all_layers(self) -> list[Layer]:
        """Return all decoder layers."""
        return [
            self.get_layer(i) for i in range(self.config.num_hidden_layers)
        ]

    def count_comm_ops_per_layer(self, layer: Layer) -> int:
        """Count communication ops in a layer."""
        return sum(1 for op in layer.operators if op.comm_after is not None)
