from collections import deque

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name


_TASKS = {
	# Craftax Classic is a faithful, accelerator-friendly implementation of Crafter.
	'crafter': ('Craftax-Classic-Symbolic-v1', True),
	'craftax-classic': ('Craftax-Classic-Symbolic-v1', True),
	'craftax': ('Craftax-Symbolic-v1', False),
}


class CraftaxWrapper(gym.Env):
	"""Gymnasium-style adapter for the functional JAX Craftax environments."""

	metadata = {'render_modes': ['rgb_array']}

	def __init__(self, cfg, env_name, classic):
		super().__init__()
		self.cfg = cfg
		self._classic = classic
		self._env = make_craftax_env_from_name(env_name, auto_reset=False)
		# `env.reset`/`env.step` are already jitted upstream; we only pin placement.
		self._device = jax.devices(cfg.craftax_device)[0]
		with jax.default_device(self._device):
			self._params = self._env.default_params.replace(
				max_timesteps=int(cfg.craftax_max_steps),
			)
		self._rng = jax.device_put(jax.random.PRNGKey(cfg.seed), self._device)
		self._state = None
		self._render = None
		self._frames = deque(maxlen=int(cfg.frame_stack))

		# TD-MPC2 treats these as episodic tasks with a categorical action space.
		self.discrete = True
		self.episodic = True

		self.action_space = gym.spaces.Discrete(self._env.num_actions)
		if cfg.obs == 'state':
			# Classic normalizes everything to [0, 1], but full Craftax packs the
			# sword/bow/armour enchantment levels in raw, where they reach 2.
			# (Upstream declares [0, 1] for both; this is the accurate bound.)
			shape = self._env.observation_space(self._params).shape
			self.observation_space = gym.spaces.Box(
				low=0., high=1. if classic else 2., shape=shape, dtype=np.float32,
			)
		elif cfg.obs == 'rgb':
			self.observation_space = gym.spaces.Box(
				low=0,
				high=255,
				shape=(3 * int(cfg.frame_stack), 64, 64),
				dtype=np.uint8,
			)
		else:
			raise ValueError('Crafter/Craftax only supports state and rgb observations.')

	@property
	def max_episode_steps(self):
		return int(self.cfg.craftax_max_steps)

	def _next_key(self):
		self._rng, key = jax.random.split(self._rng)
		return key

	def _make_renderer(self):
		if self._classic:
			from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_IMG
			from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
		else:
			from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG
			from craftax.craftax.renderer import make_craftax_pixel_renderer

		render = make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_IMG)

		def render_64(state):
			pixels = render(state)
			return jax.image.resize(pixels, (64, 64, 3), method='nearest')

		self._render = jax.jit(render_64)

	def _render_frame(self):
		if self._render is None:
			self._make_renderer()
		with jax.default_device(self._device):
			pixels = np.asarray(self._render(self._state))
		return np.clip(pixels, 0, 255).astype(np.uint8).transpose(2, 0, 1)

	def _get_obs(self, symbolic_obs=None, reset=False):
		if self.cfg.obs == 'state':
			return np.asarray(symbolic_obs, dtype=np.float32)
		frame = self._render_frame()
		if reset:
			self._frames.clear()
			for _ in range(self._frames.maxlen):
				self._frames.append(frame)
		else:
			self._frames.append(frame)
		return np.concatenate(tuple(self._frames), axis=0)

	def _terminated(self):
		"""True termination, excluding the `max_timesteps` time limit.

		Mirrors `is_game_over` in Craftax minus the `done_steps` term, so that
		hitting the time limit is reported as a truncation rather than a
		termination (TD-MPC2 bootstraps through truncations).
		"""
		dead = bool(np.asarray(self._state.player_health <= 0))
		if self._classic:
			from craftax.craftax_classic.constants import BlockType

			position = self._state.player_position
			in_lava = self._state.map[position[0], position[1]] == BlockType.LAVA.value
			return dead or bool(np.asarray(in_lava))
		return dead or bool(
			np.asarray(self._state.boss_progress >= self._env.static_env_params.num_levels - 1)
		)

	def reset(self, *, seed=None, options=None):
		del options
		if seed is not None:
			self._rng = jax.device_put(jax.random.PRNGKey(seed), self._device)
		with jax.default_device(self._device):
			obs, self._state = self._env.reset(self._next_key(), self._params)
		return self._get_obs(obs, reset=True)

	def step(self, action):
		action = int(action)
		with jax.default_device(self._device):
			obs, self._state, reward, done, raw_info = self._env.step(
				self._next_key(), self._state, jnp.asarray(action), self._params,
			)
		info = {
			key: float(np.asarray(value))
			for key, value in raw_info.items()
			if np.asarray(value).ndim == 0
		}
		info['success'] = float(
			not self._classic
			and self._state.boss_progress >= self._env.static_env_params.num_levels - 1
		)
		info['terminated'] = self._terminated()
		# Crafter/Craftax are scored by how many distinct achievements are unlocked.
		info['achievements'] = float(np.asarray(self._state.achievements).sum())
		return (
			self._get_obs(obs),
			float(np.asarray(reward)),
			bool(np.asarray(done)),
			info,
		)

	def render(self):
		# Video logging expects channel-last uint8 frames, not stacked observations.
		return self._render_frame().transpose(1, 2, 0)


def make_env(cfg):
	"""Make Crafter (Craftax Classic) or full Craftax."""
	if cfg.task not in _TASKS:
		raise ValueError('Unknown task:', cfg.task)
	if getattr(cfg, 'multitask', False):
		raise ValueError('Crafter/Craftax cannot be used in multi-task task sets.')
	env_name, classic = _TASKS[cfg.task]
	return CraftaxWrapper(cfg, env_name, classic)
