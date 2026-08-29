import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss (MNRL).
    In-batch negatives come from other positive pairs in the batch.
    Optionally accepts explicit hard negative embeddings per anchor.
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        hard_neg_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        anchor_emb: (B, D) L2 normalized
        positive_emb: (B, D) L2 normalized
        hard_neg_emb: Optional (B, D) explicit hard negative embeddings
        """
        B = anchor_emb.size(0)
        
        if hard_neg_emb is not None:
            # Candidate set includes both positive and explicit hard negative items -> (2B, D)
            candidates = torch.cat([positive_emb, hard_neg_emb], dim=0)
        else:
            # Candidate set is in-batch positive items -> (B, D)
            candidates = positive_emb
            
        # Pairwise cosine similarity matrix -> (B, candidate_count)
        sim_matrix = torch.matmul(anchor_emb, candidates.T) / self.temperature
        
        # Ground truth positive targets: for anchor i, the matching positive is at index i
        labels = torch.arange(B, device=anchor_emb.device)
        
        return self.cross_entropy(sim_matrix, labels)


class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss using Cosine Distance.
    L = max(0, margin + dist(anchor, positive) - dist(anchor, negative))
    where dist(a, b) = 1.0 - cos_sim(a, b).
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        anchor_emb: (B, D) L2 normalized
        positive_emb: (B, D) L2 normalized
        negative_emb: (B, D) L2 normalized
        """
        pos_sim = torch.sum(anchor_emb * positive_emb, dim=-1)
        neg_sim = torch.sum(anchor_emb * negative_emb, dim=-1)
        
        pos_dist = 1.0 - pos_sim
        neg_dist = 1.0 - neg_sim
        
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(losses)
