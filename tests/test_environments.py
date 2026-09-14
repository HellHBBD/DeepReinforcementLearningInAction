"""Short integration checks using real environments and their installed backends."""
import unittest
import numpy as np
import gymnasium as gym


class EnvironmentIntegration(unittest.TestCase):
    def exercise(self, env):
        try:
            observation, info = env.reset(seed=46)
            self.assertTrue(np.isfinite(np.asarray(observation)).all())
            for _ in range(5):
                observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
                self.assertTrue(np.isfinite(reward))
                if terminated or truncated:
                    env.reset()
        finally:
            env.close()

    def test_cartpole(self):
        self.exercise(gym.make('CartPole-v1', max_episode_steps=200))

    def test_freeway_ram(self):
        import ale_py
        gym.register_envs(ale_py)
        self.exercise(gym.make('ALE/Freeway-v5', obs_type='ram', frameskip=(2, 5),
                               repeat_action_probability=.25))

    def test_mario_joypad(self):
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
        self.exercise(JoypadSpace(gym.make('SuperMarioBros-v0'), COMPLEX_MOVEMENT))

    def test_doorkey(self):
        import minigrid
        from minigrid.wrappers import ImgObsWrapper
        self.exercise(ImgObsWrapper(gym.make('MiniGrid-DoorKey-5x5-v0', max_steps=400)))

    def test_battle(self):
        from magent2.environments import battle_v4
        env = battle_v4.parallel_env(map_size=30, max_cycles=5)
        try:
            observations, infos = env.reset(seed=46)
            self.assertTrue(observations)
            for _ in range(3):
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                observations, rewards, terminated, truncated, infos = env.step(actions)
                self.assertTrue(all(np.isfinite(r) for r in rewards.values()))
        finally:
            env.close()
