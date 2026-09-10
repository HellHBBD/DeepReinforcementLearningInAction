"""
python main.py
"""

import random
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym


class Agent(object):
    def __init__(
        self, env, state_space, action_space, weights=[], max_eps_length=500, trials=5
    ):
        self.env = env
        self.max_eps_length = max_eps_length
        self.trials = trials
        state_space = state_space[0]  # add batch dimension
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_size = 32

        self.weights = weights if weights else self._get_random_weights()
        self.fitness = self._get_fitness()

    def model(self, x):
        x = F.relu(torch.add(torch.mm(x, self.weights[0]), self.weights[1]))
        x = F.relu(torch.add(torch.mm(x, self.weights[2]), self.weights[3]))
        x = F.softmax(torch.add(torch.mm(x, self.weights[4]), self.weights[5]), dim=-1)
        return x

    def _get_random_weights(self):
        return [
            torch.rand(self.state_space, self.hidden_size),  # fc1 weights
            torch.rand(self.hidden_size),  # fc1 bias
            torch.rand(self.hidden_size, self.hidden_size),  # fc2 weights
            torch.rand(self.hidden_size),  # fc2 bias
            torch.rand(self.hidden_size, self.action_space),  # fc3 weights
            torch.rand(self.action_space),  # fc3 bias
        ]

    def test_agent(self, render=False):
        state = self.env.reset()[0]
        if render:
            env.render()
        total_reward, i, done = 0, 0, False
        while not done and i < self.max_eps_length:
            action = self.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            i += 1

            if render:
                env.render()

        env.close()
        return total_reward

    def _get_fitness(self):
        total_reward = 0
        for _ in range(self.trials):
            total_reward += self.test_agent()
        return total_reward / self.trials

    def get_action(self, state):
        act_prob = (
            self.model(torch.Tensor(state.reshape(1, -1))).detach().numpy()[0]
        )  # use predict api when merged
        action = np.random.choice(range(len(act_prob)), p=act_prob)
        return action

    def save(self, save_file):
        self.mod.save_params(save_file)


def cross(agent1, agent2, agent_config):
    num_params = len(agent1.weights)
    crossover_idx = np.random.randint(0, num_params)
    new_weights = agent1.weights[:crossover_idx] + agent2.weights[crossover_idx:]
    new_weights = mutate(new_weights)
    return Agent(weights=new_weights, **agent_config)


def mutate(new_weights, rate=0.001):
    for i in range(len(new_weights)):
        mask = (torch.rand(new_weights[i].size()) < rate).int()
        mutation = torch.randn(new_weights[i].size()) / 10
        new_weights[i] = new_weights[i] + mask * mutation
    return new_weights


def reproduce(agents, agent_config, generation_size):
    new_agents = []
    while len(new_agents) < generation_size:
        parents = random.choices(agents, k=2, weights=[x.fitness for x in agents])
        child1 = cross(parents[0], parents[1], agent_config)
        child2 = cross(parents[0], parents[1], agent_config)
        new_agents.extend([child1, child2])

    return new_agents


def run(n_generations, generation_size, agent_config, save_file=None, render=False):
    # Initialize population, fitness is observed on init
    agents = [Agent(**agent_config) for _ in range(generation_size)]
    agents = sorted(agents, reverse=True, key=lambda a: a.fitness)
    max_fitness = 0
    for i in range(n_generations):
        agents = reproduce(agents, agent_config, generation_size)
        agents = sorted(agents, reverse=True, key=lambda a: a.fitness)
        avg_fitness = sum([a.fitness for a in agents]) / len(agents)
        print(f"{i}, Avg: {avg_fitness:.2f}, Top: {agents[0].fitness:.2f}")
        if agents[0].fitness > max_fitness:
            max_fitness = agents[0].fitness
            # ranked_generation[0].save(args.save_file)

    final_score = agents[0].test_agent(render=render)
    print("Final fitness:", agents[0].fitness)
    print("Final score:", final_score)


if __name__ == "__main__":
    env_names = list(gym.envs.registry.keys())

    parser = ArgumentParser()
    parser.add_argument("--n_generations", default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--generation_size", default=100)
    parser.add_argument("--max_eps_length", default=500)
    parser.add_argument("--trials", default=10)
    parser.add_argument("--env", default="CartPole-v1", choices=env_names)
    parser.add_argument("--save_file")

    args = parser.parse_args()
    env = gym.make(args.env)

    agent_config = {
        "state_space": env.observation_space.shape,
        "action_space": env.action_space.n,
        "max_eps_length": args.max_eps_length,
        "trials": args.trials,
        "env": env,
    }

    run(
        args.n_generations,
        args.generation_size,
        agent_config,
        args.save_file,
        args.render,
    )
