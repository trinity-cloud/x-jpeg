import torch

from xjpeg.model import Decoder, Encoder, QTableNet, code_to_table


def test_shapes_and_ranges():
    net = QTableNet().eval()
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        tables, bits, recon = net(x)
    assert tables.shape == (2, 3, 8, 8)
    assert recon.shape == (2, 3, 256, 256)
    assert tables.min() >= 1 and tables.max() <= 255
    assert torch.equal(tables, tables.round())  # integer at eval
    assert torch.isfinite(bits)


def test_initial_tables_near_16():
    enc = Encoder().eval()
    with torch.no_grad():
        t = code_to_table(enc(torch.rand(1, 3, 256, 256)))
    assert (t - 16.0).abs().max() < 2.0  # zero-init head => bias-driven start


def test_training_grads_reach_encoder():
    net = QTableNet().train()
    x = torch.rand(2, 3, 256, 256)
    tables, bits, recon = net(x)
    loss = tables.mean() + bits * 1e-4 + recon.mean()
    loss.backward()
    g = net.encoder.head.bias.grad
    assert g is not None and torch.isfinite(g).all()


def test_conditioned_init_opens_first_step_feature_gradient():
    enc = Encoder(conditioned_init=True).train()
    x = torch.rand(2, 3, 256, 256)
    code_to_table(enc(x)).mean().backward()
    first = enc.blocks[0].down.weight.grad
    assert first is not None
    assert torch.isfinite(first).all()
    assert first.abs().sum() > 0


def test_two_table_shape_remains_supported():
    net = QTableNet(num_tables=2).eval()
    with torch.no_grad():
        tables, _, _ = net(torch.rand(1, 3, 256, 256))
    assert tables.shape == (1, 2, 8, 8)


def test_reparameterized_igdn_decoder_is_finite():
    decoder = Decoder(
        widths=(8, 8, 8, 8, 8), num_tables=3, use_igdn=True
    )
    tables = torch.rand(2, 3, 8, 8, requires_grad=True)
    recon = decoder(tables)
    assert recon.shape == (2, 3, 256, 256)
    assert torch.isfinite(recon).all()
    recon.mean().backward()
    assert tables.grad is not None
    assert torch.isfinite(tables.grad).all()


def test_compact_five_transform_autoencoder():
    net = QTableNet(
        architecture="compact", gdn_reparam=True, decoder_igdn=True
    ).train()
    x = torch.rand(1, 3, 256, 256)
    tables, bits, recon = net(x)
    assert tables.shape == (1, 3, 8, 8)
    assert recon.shape == x.shape
    assert torch.isfinite(tables).all()
    assert torch.isfinite(bits)
    assert torch.isfinite(recon).all()
    (recon.mean() + tables.mean() * 1e-4).backward()
    assert torch.isfinite(net.encoder.head.weight.grad).all()


def test_wider_compact_autoencoder_preserves_three_by_eight_bottleneck():
    net = QTableNet(
        architecture="compact", gdn_reparam=True, decoder_igdn=True,
        width_multiplier=1.5,
    ).eval()
    assert net.encoder.blocks[0].conv.out_channels == 96
    assert net.encoder.blocks[-1].conv.out_channels == 384
    assert net.decoder.stem.up.out_channels == 384
    with torch.no_grad():
        tables, bits, recon = net(torch.rand(1, 3, 256, 256))
    assert tables.shape == (1, 3, 8, 8)
    assert recon.shape == (1, 3, 256, 256)
    assert torch.isfinite(bits)


def test_compact_product_stats_are_frequency_aligned_and_conditioning():
    enc = Encoder(
        architecture="compact", dct_stats=True, conditioned_init=True
    ).eval()
    image = torch.rand(1, 3, 256, 256)
    low = torch.zeros(1, 3, 8, 8)
    high = torch.ones(1, 3, 8, 8)
    with torch.no_grad():
        low_code = enc(image, low)
        high_code = enc(image, high)
    assert low_code.shape == (1, 3, 8, 8)
    assert torch.allclose(
        high_code - low_code, enc.stats_scale / 5.0, atol=1e-6
    )
