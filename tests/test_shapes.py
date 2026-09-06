import torch
import torch.nn.functional as F
import torch.nn as nn
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    FirstViewPooling,
    ProjectionHeadWithNorm,
    ContrastiveProjectionHead,
    GlobalInfoNCE,
)
from src.train import MedAlignModel


def test_first_view_pooling_variable_views():
    B, V_max, D = 4, 2, 2048
    x = torch.randn(B, V_max, D)

    view_mask = torch.ones(B, V_max, dtype=torch.bool)
    view_mask[0, 1] = False
    view_mask[2, 1] = False

    pool = FirstViewPooling()
    out = pool(x, view_mask)

    assert out.shape == (B, D)
    # FirstViewPooling should just return the first view
    assert torch.allclose(out, x[:, 0, :])


def test_projection_head_output_shape():
    B, D_in, D_out = 16, 2048, 256
    x = torch.randn(B, D_in)

    proj = ProjectionHeadWithNorm(D_in, hidden_dim=512, output_dim=D_out, dropout=0.1)
    out = proj(x)

    assert out.shape == (B, D_out)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_projection_head_with_layernorm():
    B, D_in, D_out = 8, 768, 256
    x = torch.randn(B, D_in)

    proj = ProjectionHeadWithNorm(D_in, hidden_dim=512, output_dim=D_out, use_layernorm=True)
    out = proj(x)

    assert out.shape == (B, D_out)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_projection_head_without_layernorm():
    B, D_in, D_out = 8, 768, 256
    x = torch.randn(B, D_in)

    proj = ProjectionHeadWithNorm(D_in, hidden_dim=512, output_dim=D_out, use_layernorm=False)
    out = proj(x)

    assert out.shape == (B, D_out)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.parametrize("head_type", ["identity", "linear", "mlp"])
def test_contrastive_projection_head_output_shape(head_type):
    B, D = 8, 256
    h = F.normalize(torch.randn(B, D), dim=-1)

    head = ContrastiveProjectionHead(
        embedding_dim=D,
        head_type=head_type,
        hidden_dim=512,
        dropout=0.0,
    )
    z = head(h)

    assert z.shape == (B, D)
    assert torch.allclose(z.norm(dim=-1), torch.ones(B), atol=1e-5)


def test_global_infonce_basic():
    B, D = 32, 256
    z1 = F.normalize(torch.randn(B, D), dim=-1)
    z2 = F.normalize(torch.randn(B, D), dim=-1)

    loss_fn = GlobalInfoNCE(temperature=0.07, learnable=True)
    loss, loss_dict = loss_fn(z1, z2)

    assert loss.shape == ()
    assert loss.item() > 0
    assert "loss_i2t" in loss_dict
    assert "loss_t2i" in loss_dict
    assert "temperature" in loss_dict
    assert loss_dict["temperature"].item() > 0


def test_global_infonce_temperature_clamp():
    B, D = 16, 256
    z1 = F.normalize(torch.randn(B, D), dim=-1)
    z2 = F.normalize(torch.randn(B, D), dim=-1)

    loss_fn = GlobalInfoNCE(temperature=0.07, learnable=True, min_temp=0.01, max_temp=0.1)
    loss_fn.log_temp.data.fill_(10.0)
    _, loss_dict = loss_fn(z1, z2)
    assert loss_dict["temperature"].item() <= 0.1 + 1e-5

    loss_fn.log_temp.data.fill_(-10.0)
    _, loss_dict = loss_fn(z1, z2)
    assert loss_dict["temperature"].item() >= 0.01 - 1e-5


def test_global_infonce_fixed_temperature():
    B, D = 16, 256
    z1 = F.normalize(torch.randn(B, D), dim=-1)
    z2 = F.normalize(torch.randn(B, D), dim=-1)

    loss_fn = GlobalInfoNCE(temperature=0.07, learnable=False)
    _, loss_dict = loss_fn(z1, z2)
    assert loss_dict["temperature"].item() == pytest.approx(0.07, rel=1e-5)


def test_full_model_forward_uniform():
    B, V = 16, 2
    D_img, D_txt = 2048, 768

    image_global = torch.randn(B, V, D_img)
    view_mask = torch.ones(B, V, dtype=torch.bool)
    clinical_global = torch.randn(B, D_txt)
    augment_global = torch.randn(B, D_txt)

    model = MedAlignModel(
        image_embed_dim=D_img,
        text_embed_dim=D_txt,
        latent_dim=256,
        max_views_per_study=V,
    )

    loss, loss_dict = model(image_global, view_mask, clinical_global, augment_global)

    assert loss.shape == ()
    assert loss.item() > 0
    assert "total_loss" in loss_dict
    assert "L_text" in loss_dict
    assert "L_img_txt" in loss_dict
    assert "temperature_cross" in loss_dict
    assert "temperature_intra" in loss_dict


def test_full_model_with_separate_temperatures():
    B, V = 8, 2
    D_img, D_txt = 64, 32
    model = MedAlignModel(
        image_embed_dim=D_img,
        text_embed_dim=D_txt,
        latent_dim=16,
        hidden_dim=32,
        pool_hidden_dim=16,
        max_views_per_study=V,
        contrastive_head_type="mlp",
        contrastive_head_hidden_dim=32,
        contrastive_head_dropout=0.0,
        temperature_mode="separate",
        temperature_cross=0.05,
        temperature_intra=0.09,
        learnable_temp=False,
    )

    loss, loss_dict = model(
        torch.randn(B, V, D_img),
        torch.ones(B, V, dtype=torch.bool),
        torch.randn(B, D_txt),
        torch.randn(B, D_txt),
    )

    assert loss.shape == ()
    assert loss_dict["temperature_cross"].item() == pytest.approx(0.05, rel=1e-5)
    assert loss_dict["temperature_intra"].item() == pytest.approx(0.09, rel=1e-5)


def test_shared_temperature_reuses_one_loss_module():
    model = MedAlignModel(
        image_embed_dim=64,
        text_embed_dim=32,
        latent_dim=16,
        hidden_dim=32,
        pool_hidden_dim=16,
        contrastive_head_type="identity",
        temperature_mode="shared",
        temperature=0.07,
        learnable_temp=True,
    )

    assert model.loss_fn is not None
    assert model.cross_loss_fn is None
    assert model.intra_loss_fn is None
    assert sum(name.endswith("log_temp") for name, _ in model.named_parameters()) == 1


def test_separate_temperatures_receive_independent_gradients():
    B, V = 8, 2
    model = MedAlignModel(
        image_embed_dim=64,
        text_embed_dim=32,
        latent_dim=16,
        hidden_dim=32,
        pool_hidden_dim=16,
        max_views_per_study=V,
        contrastive_head_type="identity",
        temperature_mode="separate",
        temperature_cross=0.05,
        temperature_intra=0.09,
        learnable_temp=True,
    )

    loss, _ = model(
        torch.randn(B, V, 64),
        torch.ones(B, V, dtype=torch.bool),
        torch.randn(B, 32),
        torch.randn(B, 32),
    )
    loss.backward()

    assert model.cross_loss_fn is not None and model.intra_loss_fn is not None
    assert model.cross_loss_fn.log_temp is not model.intra_loss_fn.log_temp
    assert model.cross_loss_fn.log_temp.grad is not None
    assert model.intra_loss_fn.log_temp.grad is not None


def test_full_model_forward_variable_views():
    B = 8
    V_max = 2
    D_img, D_txt = 2048, 768

    image_global = torch.randn(B, V_max, D_img)
    clinical_global = torch.randn(B, D_txt)
    augment_global = torch.randn(B, D_txt)

    view_mask = torch.ones(B, V_max, dtype=torch.bool)
    view_mask[0, 1] = False
    view_mask[2, 1] = False
    view_mask[5, 1] = False

    model = MedAlignModel(
        image_embed_dim=D_img,
        text_embed_dim=D_txt,
        latent_dim=256,
        max_views_per_study=V_max,
    )

    loss, loss_dict = model(image_global, view_mask, clinical_global, augment_global)

    assert loss.shape == ()
    assert loss.item() > 0
    assert "total_loss" in loss_dict


def test_forward_embeddings():
    B, V = 10, 2
    D_img, D_txt = 2048, 768

    image_global = torch.randn(B, V, D_img)
    view_mask = torch.ones(B, V, dtype=torch.bool)
    clinical_global = torch.randn(B, D_txt)
    augment_global = torch.randn(B, D_txt)

    model = MedAlignModel(
        image_embed_dim=D_img,
        text_embed_dim=D_txt,
        latent_dim=256,
        max_views_per_study=V,
    )

    batch = {
        "image_global": image_global,
        "view_mask": view_mask,
        "clinical_global": clinical_global,
        "augment_global": augment_global,
    }

    h_img, h_txt, z_img, z_txt = model.forward_embeddings(batch)

    assert h_img.shape == (B, 256)
    assert h_txt.shape == (B, 256)
    assert torch.allclose(h_img.norm(dim=-1), torch.ones(B), atol=1e-5)
    assert torch.allclose(h_txt.norm(dim=-1), torch.ones(B), atol=1e-5)

    z_img_loss, z_txt_loss = model.forward_contrastive_embeddings(batch)
    assert z_img_loss.shape == (B, 256)
    assert z_txt_loss.shape == (B, 256)
    assert torch.allclose(z_img_loss.norm(dim=-1), torch.ones(B), atol=1e-5)
    assert torch.allclose(z_txt_loss.norm(dim=-1), torch.ones(B), atol=1e-5)
    assert not torch.allclose(h_img, z_img_loss)
    assert not torch.allclose(h_txt, z_txt_loss)


def test_model_with_dataparallel():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2+ GPUs for DataParallel test")

    B, V = 16, 2
    D_img, D_txt = 2048, 768

    image_global = torch.randn(B, V, D_img)
    view_mask = torch.ones(B, V, dtype=torch.bool)
    clinical_global = torch.randn(B, D_txt)
    augment_global = torch.randn(B, D_txt)

    model = MedAlignModel(
        image_embed_dim=D_img,
        text_embed_dim=D_txt,
        latent_dim=256,
        max_views_per_study=V,
    )

    model = nn.DataParallel(model).cuda()

    loss, _ = model(
        image_global.cuda(), view_mask.cuda(),
        clinical_global.cuda(), augment_global.cuda(),
    )

    assert loss.shape == ()
    assert loss.item() > 0


def test_batch_independence():
    B, V = 8, 2
    D_img, D_txt = 2048, 768

    image_global = torch.randn(B, V, D_img)
    view_mask = torch.ones(B, V, dtype=torch.bool)
    clinical_global = torch.randn(B, D_txt)
    augment_global = torch.randn(B, D_txt)

    model = MedAlignModel(
        image_embed_dim=D_img,
        text_embed_dim=D_txt,
        latent_dim=256,
        max_views_per_study=V,
    )

    batch = {
        "image_global": image_global,
        "view_mask": view_mask,
        "clinical_global": clinical_global,
        "augment_global": augment_global,
    }

    h_img, h_txt, z_img, z_txt = model.forward_embeddings(batch)

    for i in range(B):
        for j in range(i+1, B):
            img_sim = F.cosine_similarity(z_img[i:i+1], z_img[j:j+1])
            txt_sim = F.cosine_similarity(z_txt[i:i+1], z_txt[j:j+1])
            assert img_sim.item() < 0.99
            assert txt_sim.item() < 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
