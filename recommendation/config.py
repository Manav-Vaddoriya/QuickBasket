import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ExperimentConfig:
    """
    Configuration dataclass for Recommendation Model experiments and fine-tuning.
    """
    experiment_id: str = "exp01_two_tower_v3"
    description: str = "True Two-Tower (Separate Anchor & Candidate Encoders) Fine-Tuning on triplet_dataset_v3.csv"
    
    # Model Architecture
    tower_type: str = "two_tower"           # 'siamese', 'two_tower'
    base_model_name: str = "BAAI/bge-small-en-v1.5"
    use_projection_head: bool = False      # Projection MLP on top of BGE
    projection_dim: int = 384             # Output embedding dimension
    freeze_base: bool = False             # Freeze base BGE encoder parameters
    
    # Loss Function
    loss_type: str = "mnrl"               # 'mnrl', 'triplet'
    triplet_margin: float = 0.5
    mnrl_temperature: float = 0.05
    
    # Data Filtering & Sources
    include_same_category: bool = True
    positive_sources: List[str] = field(default_factory=lambda: ['positive_gold', 'positive_silver', 'positive_cross_category'])
    negative_sources: List[str] = field(default_factory=lambda: ['soft_negative', 'hard_negative'])
    
    # Training Parameters
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    random_seed: int = 42
    
    # File Paths
    data_path: str = os.path.join("sample datasets", "triplet_dataset_v3.csv")
    embeddings_pkl_path: str = os.path.join("sample datasets", "item_embeddings_50k.pkl")
    output_dir: str = "checkpoints"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
