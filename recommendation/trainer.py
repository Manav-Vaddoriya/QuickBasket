import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional

from .config import ExperimentConfig
from .evaluate import Evaluator
from .loss import MultipleNegativesRankingLoss, TripletMarginLoss

class Trainer:
    """
    Trainer Module for Siamese and Two-Tower Recommendation Models.
    Consumes batches from PairDataset or TripletDataset and computes loss gradient steps.
    """
    def __init__(
        self,
        config: ExperimentConfig,
        model: torch.nn.Module,
        train_loader: DataLoader,
        evaluator: Evaluator,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.evaluator = evaluator
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model.to(self.device)
        
        if config.loss_type == "mnrl":
            self.criterion = MultipleNegativesRankingLoss(temperature=config.mnrl_temperature)
        elif config.loss_type == "triplet":
            self.criterion = TripletMarginLoss(margin=config.triplet_margin)
        else:
            raise ValueError(f"Unknown loss_type: {config.loss_type}")
            
    def train(self) -> Dict[str, float]:
        print(f"\n--- Starting Training for Experiment: '{self.config.experiment_id}' ---")
        print(f"Device: {self.device} | Tower Architecture: {self.config.tower_type} | Loss: {self.config.loss_type}")

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        best_metric = -1.0
        best_eval_metrics = {}

        os.makedirs(self.config.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.output_dir, f"{self.config.experiment_id}_best.pt")

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0
            t0 = time.time()

            for step, batch in enumerate(self.train_loader):
                anchor_texts = batch['anchor_item']
                pos_texts = batch['positive_item']

                if hasattr(self.model, 'encode_anchor'):
                    anchor_emb = self.model.encode_anchor(anchor_texts, device=self.device)
                    pos_emb = self.model.encode_candidate(pos_texts, device=self.device)
                else:
                    anchor_emb = self.model.encode(anchor_texts, device=self.device)
                    pos_emb = self.model.encode(pos_texts, device=self.device)

                if 'negative_item' in batch:
                    neg_texts = batch['negative_item']
                    if hasattr(self.model, 'encode_candidate'):
                        neg_emb = self.model.encode_candidate(neg_texts, device=self.device)
                    else:
                        neg_emb = self.model.encode(neg_texts, device=self.device)

                    if self.config.loss_type == "mnrl":
                        loss = self.criterion(anchor_emb, pos_emb, neg_emb)
                    else:
                        loss = self.criterion(anchor_emb, pos_emb, neg_emb)
                else:
                    if self.config.loss_type == "mnrl":
                        loss = self.criterion(anchor_emb, pos_emb)
                    else:
                        raise ValueError("Triplet margin loss requires negative_item in batch! Use TripletDataset.")

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(self.train_loader))
            elapsed = time.time() - t0

            eval_metrics = self.evaluator.evaluate(self.model, device=self.device, k_values=[5, 10, 20])
            val_score = eval_metrics.get("NDCG@10", 0.0)

            print(f"Epoch {epoch}/{self.config.epochs} [{round(elapsed, 1)}s] - Train Loss: {avg_loss:.4f} | Val Hit@10: {eval_metrics.get('Hit@10', 0):.4f} | Val Recall@10: {eval_metrics.get('Recall@10', 0):.4f} | Val NDCG@10: {val_score:.4f} | Val MRR@10: {eval_metrics.get('MRR@10', 0):.4f}")

            if val_score > best_metric:
                best_metric = val_score
                best_eval_metrics = eval_metrics.copy()
                best_eval_metrics["train_loss"] = avg_loss
                best_eval_metrics["epoch"] = epoch
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  --> Saved best checkpoint to '{checkpoint_path}' (Val NDCG@10: {val_score:.4f})")

        print(f"Training Complete for '{self.config.experiment_id}'. Best Val NDCG@10: {best_metric:.4f}")
        return best_eval_metrics
