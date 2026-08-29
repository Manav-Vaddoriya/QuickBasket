from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class ProjectionHead(nn.Module):
    """
    Optional Projection Head (Linear or 2-layer MLP with LayerNorm & L2 Norm).
    Project embeddings down to 128 / 256 or keep at 384 dims.
    """
    def __init__(self, in_features: int = 384, out_features: int = 384, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features
            
        if in_features == out_features and hidden_dim == in_features:
            self.net = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_features),
                nn.LayerNorm(out_features)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)


class SiameseTower(nn.Module):
    """
    Siamese (Shared-Weight) Tower Architecture.
    Single BGE encoder shared across both anchor and candidate items.
    """
    def __init__(
        self,
        base_model_name: str = "BAAI/bge-small-en-v1.5",
        use_projection_head: bool = False,
        projection_dim: int = 384,
        freeze_base: bool = False
    ):
        super().__init__()
        self.encoder = SentenceTransformer(base_model_name)
        if hasattr(self.encoder, 'get_embedding_dimension'):
            in_dim = self.encoder.get_embedding_dimension()
        else:
            in_dim = self.encoder.get_sentence_embedding_dimension()
        
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.use_projection_head = use_projection_head
        if use_projection_head:
            self.proj = ProjectionHead(in_features=in_dim, out_features=projection_dim)
        else:
            self.proj = None

    def encode(self, texts: List[str], device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
            
        features = self.encoder.tokenize(texts)
        features = {k: v.to(device) if hasattr(v, 'to') else torch.tensor(v, device=device) for k, v in features.items()}
        outputs = self.encoder(features)
        emb = outputs['sentence_embedding']
            
        if self.use_projection_head and self.proj is not None:
            emb = self.proj(emb)
        else:
            emb = F.normalize(emb, p=2, dim=-1)
            
        return emb

    def forward(self, anchor_texts: List[str], candidate_texts: List[str]):
        anchor_emb = self.encode(anchor_texts)
        candidate_emb = self.encode(candidate_texts)
        return anchor_emb, candidate_emb


class TwoTower(nn.Module):
    """
    True Two-Tower Architecture.
    Separate anchor_encoder and candidate_encoder initialized from the same BGE checkpoint.
    """
    def __init__(
        self,
        base_model_name: str = "BAAI/bge-small-en-v1.5",
        use_projection_head: bool = False,
        projection_dim: int = 384,
        freeze_base: bool = False
    ):
        super().__init__()
        self.anchor_encoder = SentenceTransformer(base_model_name)
        self.candidate_encoder = SentenceTransformer(base_model_name)
        
        if hasattr(self.anchor_encoder, 'get_embedding_dimension'):
            in_dim = self.anchor_encoder.get_embedding_dimension()
        else:
            in_dim = self.anchor_encoder.get_sentence_embedding_dimension()

        if freeze_base:
            for param in self.anchor_encoder.parameters():
                param.requires_grad = False
            for param in self.candidate_encoder.parameters():
                param.requires_grad = False

        self.use_projection_head = use_projection_head
        if use_projection_head:
            self.anchor_proj = ProjectionHead(in_features=in_dim, out_features=projection_dim)
            self.candidate_proj = ProjectionHead(in_features=in_dim, out_features=projection_dim)
        else:
            self.anchor_proj = None
            self.candidate_proj = None

    def encode_anchor(self, texts: List[str], device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = next(self.anchor_encoder.parameters()).device
        features = self.anchor_encoder.tokenize(texts)
        features = {k: v.to(device) if hasattr(v, 'to') else torch.tensor(v, device=device) for k, v in features.items()}
        outputs = self.anchor_encoder(features)
        emb = outputs['sentence_embedding']
        
        if self.use_projection_head and self.anchor_proj is not None:
            emb = self.anchor_proj(emb)
        else:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def encode_candidate(self, texts: List[str], device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = next(self.candidate_encoder.parameters()).device
        features = self.candidate_encoder.tokenize(texts)
        features = {k: v.to(device) if hasattr(v, 'to') else torch.tensor(v, device=device) for k, v in features.items()}
        outputs = self.candidate_encoder(features)
        emb = outputs['sentence_embedding']
        
        if self.use_projection_head and self.candidate_proj is not None:
            emb = self.candidate_proj(emb)
        else:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def forward(self, anchor_texts: List[str], candidate_texts: List[str]):
        anchor_emb = self.encode_anchor(anchor_texts)
        candidate_emb = self.encode_candidate(candidate_texts)
        return anchor_emb, candidate_emb
