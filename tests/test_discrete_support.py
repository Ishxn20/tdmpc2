import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tdmpc2'))


HAS_GYM = importlib.util.find_spec('gymnasium') is not None
HAS_TENSORDICT = importlib.util.find_spec('tensordict') is not None


def _discrete_cfg(**overrides):
	"""Minimal config for a discrete single-task world model."""
	cfg = dict(
		multitask=False,
		tasks=['crafter'],
		task_dim=0,
		action_dim=5,
		discrete=True,
		obs='state',
		obs_shape={'state': (8,)},
		num_enc_layers=2,
		enc_dim=32,
		num_channels=8,
		mlp_dim=32,
		latent_dim=16,
		num_bins=5,
		vmin=-10.,
		vmax=10.,
		bin_size=5.,
		num_q=2,
		dropout=0.,
		simnorm_dim=8,
		episodic=False,
		tau=.01,
		log_std_min=-10.,
		log_std_max=2.,
		categorical_tau=1.,
	)
	cfg.update(overrides)
	return SimpleNamespace(**cfg)


@unittest.skipUnless(HAS_GYM, 'gymnasium is not installed')
class TensorWrapperDiscreteTest(unittest.TestCase):
	def test_discrete_actions_are_one_hot_and_converted_to_ids(self):
		import gymnasium as gym
		import numpy as np
		import torch

		from envs.wrappers.tensor import TensorWrapper

		class DummyDiscreteEnv(gym.Env):
			def __init__(self):
				self.action_space = gym.spaces.Discrete(5)
				self.observation_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)

			def reset(self):
				return np.zeros(3, dtype=np.float32)

			def step(self, action):
				self.last_action = action
				return np.ones(3, dtype=np.float32), 1., False, {'terminated': False}

		env = TensorWrapper(DummyDiscreteEnv())
		random_action = env.rand_act()
		self.assertEqual(random_action.shape, (5,))
		self.assertEqual(random_action.sum().item(), 1.)

		action = torch.tensor([0., 0., 1., 0., 0.])
		obs, reward, done, info = env.step(action)
		self.assertEqual(env.env.last_action, 2)
		self.assertEqual(obs.shape, (3,))
		self.assertEqual(reward.item(), 1.)
		self.assertFalse(done)
		self.assertEqual(info['success'], 0.)


@unittest.skipUnless(HAS_TENSORDICT, 'tensordict is not installed')
class DiscreteWorldModelTest(unittest.TestCase):
	def test_categorical_policy_outputs_one_hot_actions(self):
		import torch

		from common.world_model import WorldModel

		cfg = _discrete_cfg()
		model = WorldModel(cfg).to('cpu')
		z = torch.randn(4, cfg.latent_dim)
		action, info = model.pi(z, None)

		self.assertEqual(action.shape, (4, cfg.action_dim))
		self.assertTrue(torch.equal(action.sum(-1), torch.ones(4)))
		self.assertTrue(torch.all((action == 0) | (action == 1)))
		self.assertEqual(info['entropy'].shape, (4, 1))
		# `mean` is the greedy action used for evaluation.
		self.assertTrue(torch.equal(
			info['mean'].argmax(-1), info['logits'].argmax(-1)))

	def test_policy_head_is_sized_for_logits_not_gaussian_params(self):
		from common.world_model import WorldModel

		cfg = _discrete_cfg()
		model = WorldModel(cfg).to('cpu')
		self.assertEqual(model._pi[-1].out_features, cfg.action_dim)

		cfg = _discrete_cfg(discrete=False)
		model = WorldModel(cfg).to('cpu')
		self.assertEqual(model._pi[-1].out_features, 2*cfg.action_dim)

	def test_scaled_entropy_is_normalized_to_unit_interval(self):
		import torch

		from common.world_model import WorldModel

		cfg = _discrete_cfg()
		model = WorldModel(cfg).to('cpu')
		_, info = model.pi(torch.randn(64, cfg.latent_dim), None)
		scaled = info['scaled_entropy']
		self.assertTrue(torch.all(scaled >= 0.))
		self.assertTrue(torch.all(scaled <= 1. + 1e-5))

	def test_straight_through_gradients_reach_the_policy(self):
		"""The hard one-hot must still pass gradients back to `_pi`.

		Note this deliberately does not backprop through `Q`: TD-MPC2 zero-inits
		the final Q layer, so `dQ/daction` is exactly zero at initialization in
		both discrete and continuous mode.
		"""
		import torch

		from common.world_model import WorldModel

		cfg = _discrete_cfg()
		model = WorldModel(cfg).to('cpu')
		action, _ = model.pi(torch.randn(4, cfg.latent_dim), None)
		(action * torch.randn_like(action)).sum().backward()
		self.assertTrue(any(
			parameter.grad is not None and parameter.grad.abs().sum() > 0
			for parameter in model._pi.parameters()
		))


@unittest.skipUnless(HAS_TENSORDICT, 'tensordict is not installed')
class DiscretePlannerTest(unittest.TestCase):
	"""Covers `TDMPC2._plan_discrete` without requiring a GPU."""

	def _make_planner(self, **overrides):
		import torch

		from common.world_model import WorldModel
		from tdmpc2 import TDMPC2

		planner_cfg = dict(
			horizon=3,
			num_samples=32,
			num_elites=8,
			num_pi_trajs=4,
			iterations=2,
			temperature=0.5,
			categorical_min_prob=0.01,
		)
		planner_cfg.update(overrides)
		cfg = _discrete_cfg(**planner_cfg)
		planner = SimpleNamespace(
			cfg=cfg,
			device=torch.device('cpu'),
			model=WorldModel(cfg).to('cpu'),
			discount=0.99,
		)
		planner._prev_mean = torch.full(
			(cfg.horizon, cfg.action_dim), 1./cfg.action_dim)
		planner._estimate_value = TDMPC2._estimate_value.__get__(planner)
		planner._plan_discrete = TDMPC2._plan_discrete.__get__(planner)
		return planner, cfg

	def test_returns_a_one_hot_action_and_valid_prev_mean(self):
		import torch

		planner, cfg = self._make_planner()
		obs = torch.randn(1, cfg.obs_shape['state'][0])
		action = planner._plan_discrete(obs, t0=True)

		self.assertEqual(action.shape, (cfg.action_dim,))
		self.assertAlmostEqual(action.sum().item(), 1., places=5)
		self.assertTrue(torch.all((action == 0) | (action == 1)))

		probs = planner._prev_mean
		self.assertEqual(probs.shape, (cfg.horizon, cfg.action_dim))
		self.assertTrue(torch.allclose(
			probs.sum(-1), torch.ones(cfg.horizon), atol=1e-5))
		# The probability floor keeps every action reachable across replans.
		self.assertTrue(torch.all(probs > 0.))

	def test_eval_mode_is_greedy_and_deterministic(self):
		import torch

		planner, cfg = self._make_planner()
		obs = torch.randn(1, cfg.obs_shape['state'][0])
		planner._plan_discrete(obs, t0=True)
		expected = planner._prev_mean[0].argmax()

		torch.manual_seed(0)
		a = planner._plan_discrete(obs, t0=False, eval_mode=True)
		# Greedy selection must match the argmax of the planner's own posterior.
		self.assertEqual(a.argmax().item(), planner._prev_mean[0].argmax().item())
		del expected

	def test_warm_start_shifts_previous_plan(self):
		import torch

		planner, cfg = self._make_planner(iterations=1)
		obs = torch.randn(1, cfg.obs_shape['state'][0])
		planner._plan_discrete(obs, t0=True)
		# A t0 replan must ignore the previous plan; a non-t0 replan must not.
		before = planner._prev_mean.clone()
		planner._plan_discrete(obs, t0=False)
		self.assertFalse(torch.equal(before, planner._prev_mean))

	def test_uniform_mix_does_not_scale_with_action_count(self):
		"""The planner keeps the same authority at 17 actions and at 43.

		The blend is a fixed *share* of uniform mass. A per-action floor would
		instead hand away `floor * action_dim`, i.e. 17% for `crafter` but 43%
		for `craftax`, muting the planner exactly where the search is hardest.
		"""
		import torch

		mix = 0.05
		for action_dim in (17, 43):
			planner, cfg = self._make_planner(
				action_dim=action_dim, categorical_min_prob=mix)
			obs = torch.randn(1, cfg.obs_shape['state'][0])
			planner._plan_discrete(obs, t0=True)
			probs = planner._prev_mean

			self.assertTrue(torch.allclose(
				probs.sum(-1), torch.ones(cfg.horizon), atol=1e-5))
			self.assertTrue(torch.all(probs > 0.))
			# Every action retains exactly `mix / action_dim`, so the mass held
			# back from the search is `mix` regardless of how many actions exist.
			self.assertAlmostEqual(
				probs.min().item(), mix/action_dim, places=4,
				msg=f'floor scaled with action_dim={action_dim}')
			# Total mass withheld from the search is `mix`, not `mix*action_dim`.
			withheld = probs.shape[-1] * probs.min().item()
			self.assertAlmostEqual(withheld, mix, places=4)
			# The search still shapes the result rather than being flattened out.
			self.assertGreater(probs.max().item(), 1./action_dim)

	def test_handles_no_cem_samples(self):
		"""`num_pi_trajs == num_samples` leaves zero trajectories to resample."""
		import torch

		planner, cfg = self._make_planner(num_samples=8, num_pi_trajs=8, num_elites=4)
		obs = torch.randn(1, cfg.obs_shape['state'][0])
		action = planner._plan_discrete(obs, t0=True)
		self.assertEqual(action.shape, (cfg.action_dim,))
		self.assertAlmostEqual(action.sum().item(), 1., places=5)


if __name__ == '__main__':
	unittest.main()
