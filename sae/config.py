from dataclasses import dataclass
from typing import Optional, Literal
from simple_parsing import Serializable, list_field


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the full decoder weights to have unit norm."""
    
    per_block_norm: Optional[str] = None
    """Normalize decoder weights per LoRA block. One of: 'l1', 'l2', or None."""
    
    block_df_path: str = "/net/tscratch/people/plgkingak/weights2weights/weights_datasets/weight_dimensions_extended_df.pt"
    group_norms_path : str = "/net/tscratch/people/plgkingak/weights2weights/weights_datasets/group_norms_l1.pt"
    
    num_latents: int = 1000
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    multi_topk: bool = False
    """Use Multi-TopK loss."""


@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig

    batch_size: int = 8
    """Batch size measured in sequences."""
    
    eval_batch_size: int = 2
    """Batch size measured in base diffusion models."""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for SAE training."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_warmup_steps: int = 1000

    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    per_block_alpha: float = 0.0
    """Weight of the per block norm loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    feature_sampling_window: int = 100
    """Number of samples for calculating sparsity histograms."""

    hookpoints: list[str] = list_field()
    """List of hookpoints to train SAEs on."""

    layers: list[int] = list_field()
    """List of layer indices to train SAEs on."""

    layer_stride: int = 1
    """Stride between layers to train SAEs on."""

    transcode: bool = False
    """Predict the output of a module given its input."""

    distribute_modules: bool = False
    """Store a single copy of each SAE, instead of copying them across devices."""

    save_every: int = 1000
    """Save SAEs every `save_every` steps."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1

    def __post_init__(self):
        assert not (
            self.layers and self.layer_stride != 1
        ), "Cannot specify both `layers` and `layer_stride`."
