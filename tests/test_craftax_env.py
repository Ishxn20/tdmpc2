import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tdmpc2'))


HAS_CRAFTAX = importlib.util.find_spec('craftax') is not None
HAS_GYM = importlib.util.find_spec('gymnasium') is not None


@unittest.skipUnless(HAS_CRAFTAX and HAS_GYM, 'Craftax dependencies are not installed')
class CraftaxEnvironmentTest(unittest.TestCase):
	def test_crafter_symbolic_reset_and_step(self):
		import numpy as np

		from envs.craftax import make_env

		cfg = SimpleNamespace(
			task='crafter',
			obs='state',
			seed=7,
			craftax_max_steps=5,
			craftax_device='cpu',
			frame_stack=3,
		)
		env = make_env(cfg)
		obs = env.reset()
		self.assertEqual(obs.shape, env.observation_space.shape)
		self.assertEqual(obs.dtype, np.float32)

		next_obs, reward, done, info = env.step(0)
		self.assertEqual(next_obs.shape, obs.shape)
		self.assertIsInstance(reward, float)
		self.assertIsInstance(done, bool)
		self.assertIn('terminated', info)
		self.assertIn('achievements', info)


if __name__ == '__main__':
	unittest.main()
