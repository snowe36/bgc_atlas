"""Unit tests for the contrastive BGC set-encoder (synthetic, no real ESM cache)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from bgcatlas.models.encoder import (  # noqa: E402
    BGCSetEncoder,
    nt_xent_loss,
    supcon_loss,
)


def test_encoder_forward_shapes():
    model = BGCSetEncoder(input_dim=32, hidden_dim=64, embed_dim=16, proj_dim=8, pooling="attention")
    x = torch.randn(4, 10, 32)
    mask = torch.ones(4, 10, dtype=torch.bool)
    mask[:, 7:] = False
    z = model.encode(x, mask)
    p = model(x, mask, project=True)
    assert z.shape == (4, 16)
    assert p.shape == (4, 8)
    # L2-normalized
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)


@pytest.mark.parametrize("pooling", ["attention", "mean", "deepsets"])
def test_encoder_backward_all_poolings(pooling):
    model = BGCSetEncoder(input_dim=16, hidden_dim=32, embed_dim=8, proj_dim=4, pooling=pooling)
    x = torch.randn(3, 5, 16, requires_grad=False)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
            [True, True, True, True, True],
        ]
    )
    z1 = model(x, mask, project=True)
    z2 = model(x + 0.01 * torch.randn_like(x), mask, project=True)
    loss = nt_xent_loss(z1, z2, temperature=0.2)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_nt_xent_finite():
    z1 = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    z2 = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    loss = nt_xent_loss(z1, z2)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_supcon_with_shared_labels():
    z = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supcon_loss(z, labels, temperature=0.1)
    assert torch.isfinite(loss)


def test_supcon_unique_labels_returns_zero():
    z = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    labels = torch.tensor([0, 1, 2])
    loss = supcon_loss(z, labels)
    assert float(loss) == 0.0


def test_train_writes_esm_compatible_outputs(tmp_path, monkeypatch):
    """End-to-end tiny train on synthetic protein cache → output contract."""
    from bgcatlas.models import train_contrastive as tc

    rng = np.random.default_rng(0)
    n_bgc, max_p, dim = 24, 6, 32
    rows = []
    embeds = []
    emb_idx = 0
    for i in range(n_bgc):
        bgc_id = f"BGC{i:07d}"
        n_prot = int(rng.integers(2, max_p + 1))
        for j in range(n_prot):
            embeds.append(rng.normal(size=dim).astype(np.float32))
            rows.append(
                {
                    "bgc_id": bgc_id,
                    "locus_tag": f"g{j}",
                    "aa_length": int(rng.integers(50, 500)),
                    "emb_idx": emb_idx,
                }
            )
            emb_idx += 1
    protein_embeds = np.stack(embeds)
    protein_meta = pd.DataFrame(rows)

    processed = tmp_path / "processed"
    processed.mkdir()
    np.save(processed / "esm_protein_embeddings.npy", protein_embeds)
    protein_meta.to_csv(processed / "esm_protein_meta.csv", index=False)

    # feature_meta with classes + dates (half pre-cutoff)
    classes = ["PKS", "NRPS", "RiPP", "terpene"]
    feature_meta = pd.DataFrame(
        {
            "bgc_id": [f"BGC{i:07d}" for i in range(n_bgc)],
            "biosynth_class": [classes[i % len(classes)] for i in range(n_bgc)],
            "date_added": ["2020-01-01" if i < 16 else "2023-01-01" for i in range(n_bgc)],
            "n_genes": [10 + i for i in range(n_bgc)],
            "organism": ["test"] * n_bgc,
        }
    )
    feature_meta.to_parquet(processed / "feature_meta.parquet", index=False)
    # hashed features (dummy) for alignment helpers — not used by train itself
    np.save(processed / "feature_matrix.npy", rng.normal(size=(n_bgc, 20)).astype(np.float32))

    reports = tmp_path / "reports"
    artifacts = tmp_path / "artifacts"
    reports.mkdir()
    artifacts.mkdir()

    monkeypatch.setattr(tc, "PROCESSED", processed)
    monkeypatch.setattr(tc, "REPORTS", reports)
    monkeypatch.setattr(tc, "ARTIFACTS", artifacts)
    monkeypatch.setattr(tc, "ROOT", tmp_path)

    cfg = tc.TrainConfig(
        objective="simclr",
        pooling="attention",
        hidden_dim=32,
        embed_dim=16,
        proj_dim=8,
        epochs=2,
        batch_size=8,
        max_proteins=max_p,
        seed=0,
        train_cutoff="2022-09-16",
        device="cpu",
        keep_frac=0.7,
        feat_dropout=0.1,
    )
    manifest = tc.train_encoder(cfg, outdir=processed)

    emb_path = processed / "learned_embeddings.npy"
    ids_path = processed / "learned_bgc_ids.csv"
    man_path = processed / "learned_embed_manifest.json"
    ckpt_path = artifacts / "bgc_encoder.pt"

    assert emb_path.exists()
    assert ids_path.exists()
    assert man_path.exists()
    assert ckpt_path.exists()

    emb = np.load(emb_path)
    ids = pd.read_csv(ids_path)
    assert emb.shape == (n_bgc, 16)
    assert len(ids) == n_bgc
    assert set(ids.columns) >= {"bgc_id", "n_proteins_embedded"}
    assert manifest["train_cutoff"] == "2022-09-16"
    assert manifest["n_train_bgcs"] == 16
    assert "learned_" in manifest["representation_label"]
    # embeddings should be roughly unit-norm
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
