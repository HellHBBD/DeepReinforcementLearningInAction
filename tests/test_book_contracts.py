"""Fast checks for mathematical/API regressions; full learning runs are separate."""
import ast
import json
import pathlib
import sys
import textwrap
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)


def definitions(chapter, cells, **extra):
    notebook = json.loads((ROOT / f"Chapter {chapter}/Ch{chapter}_book.ipynb").read_text())
    scope = dict(torch=torch, np=np, nn=nn, F=F, **extra)
    for index in cells:
        tree = ast.parse("".join(notebook["cells"][index]["source"]))
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, f"chapter{chapter}-cell{index}", "exec"), scope)
    return scope


class BookContracts(unittest.TestCase):
    def test_printed_gridworld_reward_and_blocked_move(self):
        from Environments.Gridworld import Gridworld
        game = Gridworld()
        self.assertEqual(game.reward(), -1)
        self.assertEqual(game.makeMove("u"), (-1, 0))
        self.assertEqual(game.board.components["Player"].pos, (0, 3))
        game.board.components["Player"].pos = game.board.components["Pit"].pos
        self.assertEqual(game.reward(), -10)
        game.board.components["Player"].pos = game.board.components["Goal"].pos
        self.assertEqual(game.reward(), 10)

    def test_negative_terminal_distribution_does_not_bootstrap(self):
        s = definitions(7, [7, 18])
        support = torch.linspace(-10, 10, 51)
        uniform = torch.full((2, 3, 51), 1 / 51)
        result = s["get_target_dist"](uniform, torch.tensor([0, 1]), torch.tensor([-10., -1.]), support,
                                       terminal_batch=torch.tensor([True, False]))
        self.assertEqual(result[0, 0, 0].item(), 1)
        self.assertEqual(torch.count_nonzero(result[0, 0]).item(), 1)
        self.assertGreater(torch.count_nonzero(result[1, 1]).item(), 1)
        torch.testing.assert_close(result.sum(-1), torch.ones(2, 3))

    def test_distribution_targets_preserve_untaken_actions(self):
        s = definitions(7, [7, 18])
        current = torch.softmax(torch.arange(306, dtype=torch.float32).reshape(2, 3, 51) / 10, -1)
        next_state = torch.full((2, 3, 51), 1 / 51)
        result = s["get_target_dist"](next_state, torch.tensor([0, 1]), torch.tensor([-10., 10.]),
                                      torch.linspace(-10, 10, 51), terminal_batch=torch.ones(2, dtype=torch.bool),
                                      current_dist_batch=current)
        torch.testing.assert_close(result[0, 1:], current[0, 1:])
        torch.testing.assert_close(result[1, [0, 2]], current[1, [0, 2]])

    def test_distribution_loss_is_finite_at_zero_probability(self):
        loss = definitions(7, [20])["lossfn"](torch.tensor([[[0., 1.]]]), torch.tensor([[[1., 0.]]]))
        self.assertTrue(torch.isfinite(loss).all())

    def test_mario_policy_uses_action_axis_and_entire_action_space(self):
        policy = definitions(8, [12])["policy"]
        q = torch.tensor([[4., 0., 0.]])
        with patch.object(torch, "multinomial", return_value=torch.tensor([0])) as sample:
            policy(q)
        expected = torch.tensor([np.e / (np.e + 2), 1 / (np.e + 2), 1 / (np.e + 2)], dtype=torch.float32)
        torch.testing.assert_close(sample.call_args.args[0], expected)
        with patch.object(torch, "randint", return_value=torch.tensor([11])) as choose:
            self.assertEqual(policy(torch.zeros(1, 12), eps=1).item(), 11)
        self.assertEqual(choose.call_args.kwargs["high"], 12)

    def test_mario_target_is_independent_of_other_batch_members(self):
        s = definitions(8, [26])
        class Replay:
            def __init__(self):
                self.next = torch.tensor([[1., 2.], [3., 4.]])
            def get_batch(self):
                return torch.ones(2, 2), torch.zeros(2, dtype=torch.long), torch.zeros(2), self.next, torch.zeros(2)
        replay = Replay()
        seen = []
        s.update(replay=replay, params={"eta": 1., "gamma": .2}, Qmodel=lambda x: x,
                 ICM=lambda *args: (torch.zeros(2, 1), torch.zeros(2, 1)),
                 qloss=lambda predicted, target: seen.append(target.clone()) or ((predicted-target)**2).mean())
        s["minibatch_train"](False)
        replay.next[1] = 10000
        s["minibatch_train"](False)
        torch.testing.assert_close(seen[0][0], seen[1][0])

    def test_mario_replay_retains_terminal_flag_and_single_action_batch(self):
        s = definitions(8, [14, 16], shuffle=lambda x: None)
        replay = s["ExperienceReplay"](N=2, batch_size=1)
        replay.add_memory(torch.zeros(1, 3, 42, 42), 11, 0, torch.ones(1, 3, 42, 42), True)
        batch = replay.get_batch()
        self.assertEqual(batch[-1].item(), 1)
        self.assertEqual(tuple(s["Fnet"]()(torch.zeros(1, 288), torch.tensor([[11]])).shape), (1, 288))

    def test_battle_updates_caller_and_only_selected_targets(self):
        s = definitions(9, [44])
        s["qfunc"] = lambda state, param, layers: param
        param = torch.zeros(2, requires_grad=True)
        replay = [(torch.zeros(1), torch.tensor(a), torch.tensor(r), torch.zeros(1), torch.zeros(2))
                  for a, r in [(0, 1.), (1, -1.)]]
        with patch.object(np.random, "randint", return_value=np.array([0, 1])):
            s["train"](2, replay, param, layers=[], gamma=0, lr=.1)
        torch.testing.assert_close(param, torch.tensor([.2, -.2]))
        self.assertIsNone(param.grad)

    def test_softmax_is_stable_for_book_overflow_example(self):
        softmax = definitions(9, [21])["softmax"]
        p = softmax(torch.tensor([10., 5., 90.]), temp=.1)
        self.assertTrue(torch.isfinite(p).all())
        self.assertEqual(p.argmax().item(), 2)
        self.assertEqual(p.sum().item(), 1)

    def test_rotation_is_applied_and_single_image_is_supported(self):
        import torchvision as TV
        s = definitions(10, [2, 6], TV=TV)
        np.random.seed(1)
        affine = TV.transforms.functional.affine
        with patch.object(TV.transforms.functional, "affine", wraps=affine) as transform:
            out = s["prepare_images"](torch.randint(0, 256, (8, 28, 28), dtype=torch.uint8), rot=30)
        self.assertTrue(any(call.args[1] != 0 for call in transform.call_args_list))
        self.assertTrue(all(isinstance(call.args[1], int) for call in transform.call_args_list))
        with torch.no_grad():
            self.assertEqual(tuple(s["RelationalModule"]()(out[:1, None]).shape), (1, 10))

    def test_doorkey_finite_horizon_target_and_actual_limit(self):
        import gymnasium as gym
        import minigrid
        from minigrid.wrappers import ImgObsWrapper
        target = definitions(10, [20])["get_qtarget_ddqn"](torch.tensor([100., 100.]), torch.tensor([-0.01, 1.]), .99, torch.ones(2))
        torch.testing.assert_close(target, torch.tensor([-0.01, 1.]))
        from einops import rearrange
        scope = definitions(10, [18, 20], rearrange=rearrange, gym=gym, ImgObsWrapper=ImgObsWrapper)
        notebook = json.loads((ROOT / "Chapter 10/Ch10_book.ipynb").read_text())
        setup = "".join(notebook["cells"][24]["source"]).split("for i in range(epochs):")[0]
        exec(setup, scope)
        env = scope["env"]
        self.assertEqual(env.unwrapped.max_steps, 400)
        try:
            env.reset(seed=0)
            for _ in range(399):
                _, _, terminated, truncated, _ = env.step(0)
                self.assertFalse(terminated or truncated)
            self.assertTrue(env.step(0)[3])
        finally:
            env.close()


    def test_double_dqn_selects_at_next_state(self):
        s = definitions(10, [20])
        n = json.loads((ROOT / "Chapter 10/Ch10_book.ipynb").read_text())
        source = "".join(n["cells"][24]["source"])
        start = source.index("        q_pred =")
        end = source.index("        loss =", start)
        target_code = textwrap.dedent(source[start:end])
        s.update(GWagent=lambda x: x, Tnet=lambda x: torch.tensor([[3., 7.], [5., 9.]]),
                 state_batch=torch.tensor([[10., 1.], [1., 10.]]),
                 state2_batch=torch.tensor([[1., 10.], [10., 1.]]),
                 reward_batch=torch.tensor([1., 2.]), done_batch=torch.zeros(2), gamma=.5)
        exec(target_code, s)
        torch.testing.assert_close(s["targets"], torch.tensor([4.5, 4.5]))

    def test_standalone_agent_owns_environment_and_can_save(self):
        import gymnasium as gym
        Agent = runpy.run_path(str(ROOT / "Chapter 6/main.py"))["Agent"]
        env = gym.make("CartPole-v1", max_episode_steps=2)
        try:
            a = Agent(env, (4,), 2, max_eps_length=2, trials=1)
            self.assertEqual(a.fitness, 2)
            with tempfile.TemporaryDirectory() as d:
                filename = pathlib.Path(d) / "weights.pth"
                a.save(filename)
                self.assertEqual(len(torch.load(filename, weights_only=True)), 6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
