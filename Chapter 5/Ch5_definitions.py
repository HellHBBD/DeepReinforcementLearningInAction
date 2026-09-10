import pickle
from typing import Optional

import torch
import numpy as np
import gymnasium as gym
import torch.multiprocessing as torchmp  # A
from torch import nn
from torch import optim
from torch.nn import functional as F


class ActorCritic(nn.Module):  # B
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=0)  # C
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))  # D
        return actor, critic  # E


def worker(t, worker_model, counter, params, logs):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())  # A
    eplen_acc = 0
    for i in range(params["epochs"]):
        worker_opt.zero_grad()
        if params["n_steps"] is None:
            values, logprobs, rewards, G = run_episode(worker_env, worker_model)  # B
            eplen = len(rewards)
        else:
            values, logprobs, rewards, G, eplen, eplen_acc = run_episode_n(
                worker_env, worker_model, params["n_steps"], eplen_acc
            )
        actor_loss, critic_loss = update_params(
            worker_opt, values, logprobs, rewards, G
        )  # C
        counter.value = counter.value + 1  # D
        if eplen != 0:
            logs.append(
                (
                    t,
                    i,
                    eplen,
                    round(actor_loss.detach().mean().item(), 2),
                    round(critic_loss.detach().mean().item(), 2),
                )
            )


def run_episode(worker_env, worker_model):
    state_, _ = worker_env.reset()
    state = torch.from_numpy(state_).float()  # A
    values, logprobs, rewards = [], [], []  # B
    done = False
    j = 0
    while not done:  # C
        j += 1
        policy, value = worker_model(state)  # D
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()  # E
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, terminated, truncated, _ = worker_env.step(action.detach().numpy())
        done = terminated or truncated
        state = torch.from_numpy(state_).float()
        if done:  # F
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)
    G = torch.Tensor([0.0])  # You will see this later in this chapter
    return values, logprobs, rewards, G


def update_params(worker_opt, values, logprobs, rewards, G, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)  # A
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = G
    for r in range(rewards.shape[0]):  # B
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)
    actor_loss = -1 * logprobs * (Returns - values.detach())  # C
    critic_loss = torch.pow(values - Returns, 2)  # D
    loss = actor_loss.sum() + clc * critic_loss.sum()  # E
    worker_opt.zero_grad()
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss


def train_actor_critic(ac_model, params):
    processes = []  # C
    counter = torchmp.Value("i", 0)  # D
    logs = torchmp.Manager().list()
    for i in range(params["n_workers"]):
        p = torchmp.Process(
            target=worker, args=(i, ac_model, counter, params, logs)
        )  # E
        p.start()
        processes.append(p)
    for p in processes:  # F
        p.join()
    for p in processes:  # G
        p.terminate()

    print(counter.value)  # H
    failed = False
    for i, p in enumerate(processes):
        if p.exitcode != 0:
            failed = True
            print(f"Process {i} failed with exit code {p.exitcode}")
        else:
            print(f"Process {i} success. {p.exitcode}")

    assert not failed, "At least one process failed, model might not learn anything!"

    with open("logs.pkl", "wb") as f:
        pickle.dump(list(logs), f)


def run_episode_n(worker_env, worker_model, N_steps, eplen_acc):
    state_ = np.array(worker_env.unwrapped.state, dtype=np.float32)
    state = torch.from_numpy(state_).float()
    values, logprobs, rewards = [], [], []
    done = False
    j = 0
    eplen = 0
    while j < N_steps and not done:  # B
        j += 1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, terminated, truncated, _ = worker_env.step(action.detach().numpy())
        done = terminated or truncated
        state = torch.from_numpy(state_).float()
        if done:
            reward = -10
            worker_env.reset()
            eplen = eplen_acc
            eplen_acc = 0
        else:  # C
            reward = 1.0
            eplen_acc += 1
        rewards.append(reward)

    if done:
        G = torch.Tensor([0.0])  # A
    else:
        G = value.detach()
    return values, logprobs, rewards, G, eplen, eplen_acc
