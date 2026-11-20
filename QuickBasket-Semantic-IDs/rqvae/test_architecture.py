"""
Independent Architecture & Gradient Flow Verification for RQ-VAE.

Performs rigorous unit tests on:
1. Encoder forward pass & output shapes
2. Residual Quantizer: residual progression, discrete codes, and STE behavior
3. Decoder forward pass & reconstruction shape
4. End-to-end forward pass on real product embeddings
5. Multi-task loss calculation (Recon + Codebook + Commitment)
6. Complete autograd backward pass & gradient flow checks across all components
7. Codebook diagnostics and perplexity calculations
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local imports work cleanly
sys.path.append(os.path.abspath("QuickBasket-Semantic-IDs"))

from rqvae.encoder import RQVAEEncoder
from rqvae.decoder import RQVAEDecoder
from rqvae.quantizer import ResidualQuantizer
from rqvae.model import RQVAE
from rqvae.losses import RQVAELoss


def test_encoder():
    print("\n--- Test 1: RQ-VAE Encoder ---")
    batch_size = 32
    input_dim = 384
    latent_dim = 64
    
    encoder = RQVAEEncoder(input_dim=input_dim, hidden_dims=[512, 256], latent_dim=latent_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    
    z = encoder(dummy_input)
    assert z.shape == (batch_size, latent_dim), f"Expected shape {(batch_size, latent_dim)}, got {z.shape}"
    assert not torch.isnan(z).any(), "NaN detected in encoder output"
    print(f"[PASS] Encoder forward pass successful. Output shape: {z.shape}")


def test_quantizer():
    print("\n--- Test 2: Residual Quantizer & STE ---")
    batch_size = 32
    latent_dim = 64
    codebook_size = 128
    num_levels = 3
    
    quantizer = ResidualQuantizer(num_levels=num_levels, codebook_size=codebook_size, latent_dim=latent_dim)
    z = torch.randn(batch_size, latent_dim, requires_grad=True)
    
    quantized_st, q_sum, codes, quantized_list, residuals_list = quantizer(z)
    
    # 1. Output shapes
    assert quantized_st.shape == (batch_size, latent_dim)
    assert q_sum.shape == (batch_size, latent_dim)
    assert len(codes) == num_levels
    assert len(quantized_list) == num_levels
    assert len(residuals_list) == num_levels
    
    # 2. Codes range check
    for lvl, c in enumerate(codes):
        assert c.shape == (batch_size,)
        assert (c >= 0).all() and (c < codebook_size).all(), f"Codes out of bounds at level {lvl}"
        print(f"       Level {lvl} code range: [{c.min().item()}, {c.max().item()}] (Sample: {c[:4].tolist()})")
        
    # 3. Sum of code vectors check
    expected_sum = sum(quantized_list)
    assert torch.allclose(q_sum, expected_sum, atol=1e-5), "q_sum does not equal sum of quantized vectors"
    
    # 4. Straight-Through Estimator forward equality
    assert torch.allclose(quantized_st, q_sum, atol=1e-6), "STE forward output must strictly equal q_sum"
    
    # 5. Residual progression check: norm of residual should strictly decrease
    norm_r0 = torch.norm(residuals_list[0], dim=-1).mean().item()
    norm_r1 = torch.norm(residuals_list[1], dim=-1).mean().item()
    norm_r2 = torch.norm(residuals_list[2], dim=-1).mean().item()
    print(f"       Mean residual norm progression: r0={norm_r0:.4f} -> r1={norm_r1:.4f} -> r2={norm_r2:.4f}")
    assert norm_r1 < norm_r0, f"Residual norm did not decrease at level 1: {norm_r1} >= {norm_r0}"
    assert norm_r2 < norm_r1, f"Residual norm did not decrease at level 2: {norm_r2} >= {norm_r1}"
    
    print("[PASS] Residual Quantizer & STE validated.")


def test_decoder():
    print("\n--- Test 3: RQ-VAE Decoder ---")
    batch_size = 32
    latent_dim = 64
    output_dim = 384
    
    decoder = RQVAEDecoder(latent_dim=latent_dim, hidden_dims=[256, 512], output_dim=output_dim)
    dummy_q = torch.randn(batch_size, latent_dim)
    
    x_hat = decoder(dummy_q)
    assert x_hat.shape == (batch_size, output_dim), f"Expected shape {(batch_size, output_dim)}, got {x_hat.shape}"
    assert not torch.isnan(x_hat).any(), "NaN detected in decoder output"
    print(f"[PASS] Decoder forward pass successful. Output shape: {x_hat.shape}")


def test_end_to_end_and_gradients():
    print("\n--- Test 4: End-to-End Pipeline & Gradient Flow ---")
    batch_size = 64
    input_dim = 384
    latent_dim = 64
    codebook_size = 128
    num_levels = 3
    beta = 0.25
    
    model = RQVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        codebook_size=codebook_size,
        num_levels=num_levels
    )
    criterion = RQVAELoss(beta=beta)
    
    # Try using real embeddings if available, else synthetic
    emb_path = "QuickBasket-Semantic-IDs/embeddings/product_embeddings.npy"
    if os.path.exists(emb_path):
        real_embs = np.load(emb_path)
        batch_x = torch.tensor(real_embs[:batch_size], dtype=torch.float32)
        print("       (Testing with real QuickBasket embeddings batch)")
    else:
        batch_x = F.normalize(torch.randn(batch_size, input_dim), p=2, dim=-1)
        print("       (Testing with synthetic normalized batch)")
        
    # Forward Pass
    x_hat, quantized_st, codes, quantized_list, residuals_list = model(batch_x)
    assert x_hat.shape == batch_x.shape
    
    # Loss Calculation
    loss, loss_dict = criterion(
        x=batch_x,
        x_hat=x_hat,
        residuals_list=residuals_list,
        quantized_list=quantized_list
    )
    
    print(f"       Loss breakdown: Total={loss_dict['loss_total']:.6f}, Recon={loss_dict['loss_recon']:.6f}, "
          f"Codebook={loss_dict['loss_codebook']:.6f}, Commit={loss_dict['loss_commitment']:.6f}")
    assert loss_dict["loss_recon"] > 0.0
    assert loss_dict["loss_codebook"] > 0.0
    assert loss_dict["loss_commitment"] > 0.0
    
    # Backward Pass
    loss.backward()
    
    # Check 1: Encoder Gradients
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0, "Encoder received no gradients!"
    enc_grad_norm = torch.norm(torch.stack([torch.norm(g) for g in enc_grads])).item()
    print(f"       Encoder Gradient Norm: {enc_grad_norm:.6f} (Flows through STE successfully)")
    assert enc_grad_norm > 0.0, "Encoder gradient norm is zero!"
    
    # Check 2: Codebook Gradients
    cb_grad = model.quantizer.codebooks.grad
    assert cb_grad is not None, "Codebook received no gradients!"
    cb_grad_norm = torch.norm(cb_grad).item()
    print(f"       Codebook Gradient Norm: {cb_grad_norm:.6f} (Updated by Codebook Loss)")
    assert cb_grad_norm > 0.0, "Codebook gradient norm is zero!"
    
    # Check 3: Decoder Gradients
    dec_grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert len(dec_grads) > 0, "Decoder received no gradients!"
    dec_grad_norm = torch.norm(torch.stack([torch.norm(g) for g in dec_grads])).item()
    print(f"       Decoder Gradient Norm: {dec_grad_norm:.6f} (Updated by Reconstruction Loss)")
    assert dec_grad_norm > 0.0, "Decoder gradient norm is zero!"
    
    print("[PASS] End-to-end model and gradient flow successfully verified.")


def test_codebook_diagnostics():
    print("\n--- Test 5: Codebook Perplexity & Utilization Metrics ---")
    quantizer = ResidualQuantizer(num_levels=3, codebook_size=128, latent_dim=64)
    dummy_codes = [
        torch.randint(0, 128, (1000,)),
        torch.randint(0, 80, (1000,)),
        torch.randint(0, 40, (1000,))
    ]
    metrics = quantizer.compute_codebook_metrics(dummy_codes)
    for m in metrics:
        print(f"       Level {m['level']}: Active {m['active_codes']}/128 ({m['utilization_pct']}%) | Perplexity: {m['perplexity']}")
        assert m["active_codes"] > 0
        assert m["perplexity"] > 0
    print("[PASS] Codebook metrics computation verified.")


def run_all_tests():
    print("=" * 70)
    print("STAGE 4: RQ-VAE ARCHITECTURE UNIT TESTING SUITE")
    print("=" * 70)
    test_encoder()
    test_quantizer()
    test_decoder()
    test_end_to_end_and_gradients()
    test_codebook_diagnostics()
    print("\n" + "=" * 70)
    print("ALL STAGE 4 ARCHITECTURE UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

