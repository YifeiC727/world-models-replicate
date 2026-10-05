import numpy as np
import torch

from world_models.env import PixelCatcher
from world_models.carracing_data import EpisodeChunkSampler, FrameDataset, split_episodes
from world_models.carracing_env import preprocess_frame
from world_models.models import VAE, MDNRNN, LinearController


def test_environment_is_seeded_and_has_expected_api():
    a, b = PixelCatcher(seed=7), PixelCatcher(seed=7)
    oa, _ = a.reset(7); ob, _ = b.reset(7)
    assert oa.shape == (64, 64, 3) and oa.dtype == np.uint8
    assert np.array_equal(oa, ob)
    for action in [0, 1, 2]:
        oa, reward, terminated, truncated, _ = a.step(action)
        assert np.isfinite(reward) and not truncated


def test_model_shapes_and_finite_losses():
    vae = VAE(latent_size=8); x = torch.rand(2, 3, 64, 64)
    recon, mu, logvar = vae(x); loss, _ = vae.loss(recon, x, mu, logvar)
    assert recon.shape == x.shape and mu.shape == (2, 8) and torch.isfinite(loss)
    mdn = MDNRNN(latent_size=8, hidden_size=16, mixtures=3)
    z = torch.randn(2, 5, 8); actions = torch.randint(0, 3, (2, 5))
    assert torch.isfinite(mdn.nll(z, actions, torch.randn_like(z)))


def test_controller_flat_parameter_round_trip():
    c = LinearController(latent_size=8, hidden_size=16)
    values = np.arange(c.parameter_count, dtype=np.float32) / 100
    c.set_flat_parameters(values)
    actual = torch.cat([p.detach().flatten() for p in c.parameters()]).numpy()
    assert np.allclose(values, actual)


def test_paper_carracing_shapes_and_preprocessing():
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    frame[:84] = 10
    frame[84:] = 200
    processed = preprocess_frame(frame, invert=True)
    assert processed.shape == (64, 64, 3)
    assert processed.dtype == np.uint8
    assert np.all(processed == 245)  # HUD was cropped before resizing.
    controller = LinearController(latent_size=32, hidden_size=256, action_size=3)
    assert controller.parameter_count == 867


def test_continuous_mdn_shapes():
    model = MDNRNN(latent_size=32, action_size=3, hidden_size=16,
                   mixtures=5, discrete_actions=False)
    z = torch.randn(2, 7, 32)
    actions = torch.randn(2, 7, 3)
    logits, means, log_stds, _ = model(z, actions)
    assert logits.shape == means.shape == log_stds.shape == (2, 7, 32, 5)
    assert torch.isfinite(model.nll(z, actions, torch.randn_like(z)))


def test_paper_parameter_counts():
    vae = VAE(32)
    mdn = MDNRNN(32, 3, 256, 5, discrete_actions=False)
    assert sum(p.numel() for p in vae.parameters() if p.requires_grad) == 4_348_547
    assert sum(p.numel() for p in mdn.parameters() if p.requires_grad) == 422_368


def test_episode_chunk_sampler_covers_all_frames_once(tmp_path):
    for episode, length in enumerate([5, 7, 3, 6]):
        np.savez_compressed(tmp_path / f"episode_{episode:05d}.npz",
                            observations=np.zeros((length, 64, 64, 3), dtype=np.uint8))
    dataset = FrameDataset(tmp_path, cache_size=2)
    order = list(EpisodeChunkSampler(dataset, chunk_size=2, seed=0))
    assert sorted(order) == list(range(len(dataset)))
    episodes = np.searchsorted(dataset.cumulative, order, side="right") - 1
    seen, consumed = set(), 0
    for episode in episodes:  # every chunk exhausts its episodes before the next begins
        seen.add(int(episode)); consumed += 1
        if consumed == sum(dataset.lengths[e] for e in seen):
            assert len(seen) <= 2; seen, consumed = set(), 0
    assert not seen


def test_episode_split_is_disjoint_complete_and_deterministic():
    names = [f"episode_{i:05d}.npz" for i in range(50)]
    splits = split_episodes(names, .1, .1, seed=3)
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [40, 5, 5]
    assert sorted(sum(splits.values(), [])) == names
    assert splits == split_episodes(list(reversed(names)), .1, .1, seed=3)


def test_collection_is_resumable_and_independent_of_workers(tmp_path):
    from world_models.carracing_data import collect_rollouts
    serial = collect_rollouts(tmp_path / "a", episodes=2, seed=5, max_steps=20)
    parallel = collect_rollouts(tmp_path / "b", episodes=2, seed=5, max_steps=20, workers=2)
    assert serial == parallel
    for name in ("episode_00000.npz", "episode_00001.npz"):
        with np.load(tmp_path / "a" / name) as a, np.load(tmp_path / "b" / name) as b:
            assert np.array_equal(a["observations"], b["observations"])
    before = (tmp_path / "a" / "episode_00000.npz").stat().st_mtime_ns
    assert collect_rollouts(tmp_path / "a", episodes=2, seed=5, max_steps=20) == serial
    assert (tmp_path / "a" / "episode_00000.npz").stat().st_mtime_ns == before
    assert not list((tmp_path / "a").glob("tmp_*"))


def test_reference_random_policy_actions_are_finite_and_in_range():
    from world_models.carracing_env import ReferenceRandomPolicy
    rng = np.random.default_rng(0)
    for seed in range(5):
        policy = ReferenceRandomPolicy(seed)
        for _ in range(3):
            action = policy(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
            assert np.isfinite(action).all()
            assert -1 <= action[0] <= 1 and 0 <= action[1] <= 1 and 0 <= action[2] <= 1


def test_pixelcatcher_wind_and_fog():
    from world_models.env import PixelCatcherConfig
    assert PixelCatcher(seed=3).wind == 0.0  # defaults unchanged
    env = PixelCatcher(PixelCatcherConfig(wind_range=0.08, fog_y=25), seed=3)
    assert 0 < abs(env.wind) <= 0.08
    ball = np.array([245, 210, 45], dtype=np.uint8)
    for _ in range(40):
        frame, _, _, _, info = env.step(1)
        assert info["wind"] == env.wind
        visible = (frame == ball).all(-1).any()
        assert visible == (int(round(env.ball_y)) < 25)


def test_catcher_action_mapping_matches_reference_thresholds():
    from world_models.catcher_pipeline import to_action
    out = to_action(torch.atanh(torch.tensor([[-.5], [-.2], [0.], [.2], [.5]])))
    assert out.tolist() == [0, 1, 1, 1, 2]
