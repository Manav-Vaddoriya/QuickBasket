import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from recommendation.config import ExperimentConfig
from recommendation.data_loader import load_and_prepare_data, PairDataset, TripletDataset
from recommendation.model import TwoTower
from recommendation.evaluate import Evaluator
from recommendation.trainer import Trainer
from recommendation.inference import ItemRecommender

def execute_training_pipeline():
    print("=" * 80)
    print("EXECUTING TWO-TOWER RECOMMENDATION MODEL FINE-TUNING ON triplet_dataset_v3.csv")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    # 1. Dataclass Config with updated model checkpoint name
    cfg = ExperimentConfig(
        experiment_id="two_tower_model_v3",
        description="True Two-Tower (Separate Anchor & Candidate Encoders) Fine-Tuning on triplet_dataset_v3.csv",
        tower_type="two_tower",
        loss_type="mnrl",
        data_path=os.path.join("sample datasets", "triplet_dataset_v3.csv"),
        embeddings_pkl_path=os.path.join("sample datasets", "item_embeddings_50k.pkl"),
        batch_size=64,
        epochs=1,
        learning_rate=2e-5
    )

    # 2. Prepare Data
    train_df, val_df, test_df, catalog_df, item_to_cat = load_and_prepare_data(
        data_path=cfg.data_path,
        seed=cfg.random_seed
    )

    sample_train_df = train_df.head(10000).reset_index(drop=True)
    sample_val_df = val_df.head(1000).reset_index(drop=True)
    sample_test_df = test_df.head(1000).reset_index(drop=True)

    train_ds = PairDataset(sample_train_df)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    val_evaluator = Evaluator(sample_val_df, catalog_df)
    test_evaluator = Evaluator(sample_test_df, catalog_df)

    # 3. Instantiate TwoTower Model Architecture
    model = TwoTower(
        base_model_name=cfg.base_model_name,
        use_projection_head=cfg.use_projection_head,
        projection_dim=cfg.projection_dim
    )

    # 4. Fine-Tune Model
    trainer = Trainer(cfg, model, train_loader, val_evaluator, device=device)
    best_val_metrics = trainer.train()

    # 5. Load Best Checkpoint & Evaluate Test Set
    checkpoint_path = os.path.join(cfg.output_dir, f"{cfg.experiment_id}_best.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"\nLoaded best trained checkpoint from '{checkpoint_path}'")

    print("\n--- Evaluating Fine-Tuned Model on Test Set ---")
    test_metrics = test_evaluator.evaluate(model, device=device, k_values=[5, 10, 20])
    for k, v in test_metrics.items():
        print(f"  Test {k}: {v:.4f}")

    # 6. Test Model Predictor Interface with Trained Model
    print("\n--- Model Recommendation Predictor Interface ---")
    recommender = ItemRecommender(model, catalog_df)
    sample_product = "Mother Dairy Milk"
    predictions = recommender.predict(sample_product, top_k=10)
    
    print(f"\nInput Product: '{sample_product}'")
    print(predictions.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"SUCCESS: TRAINING PIPELINE EXECUTED & CHECKPOINT SAVED TO '{checkpoint_path}'!")
    print("=" * 80)

if __name__ == '__main__':
    execute_training_pipeline()
