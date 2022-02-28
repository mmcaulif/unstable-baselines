import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
from jax.lax import stop_gradient
import haiku as hk
from collections import deque
from typing import NamedTuple
import optax
import gym
import random
import os
from functools import partial

rng = jax.random.PRNGKey(42)

@jax.vmap
def noisy_action(a_t):
    noise = (jax.random.normal(rng, shape=a_t.shape) * 0.2).clip(-0.5, 0.5)
    return (a_t + noise).clip(-env.action_space.low[0],env.action_space.high[0])

@hk.transform
def q_val(S, A):
    SA = jnp.concatenate([S, A])

    q1_seq = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(1),
    ])
    q2_seq = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(1),
    ])
    return q1_seq(SA), q2_seq(SA)


replay_buffer = deque(maxlen=100000)
env = gym.make('LunarLanderContinuous-v2')
avg_r = deque(maxlen=10)

critic_dims = (jnp.zeros(env.observation_space.shape[0]), jnp.zeros(env.action_space.shape[0]))
#print(*critic_dims)
q_params = q_val.init(rng, *critic_dims)
q_params_t = hk.data_structures.to_immutable_dict(q_params)
q_forward = hk.without_apply_rng(q_val)
q_optimizer = optax.adam(1e-3)
q_optim_state = q_optimizer.init(q_params)

s_t = env.reset()
r_sum = 0

for i in range(10):
    a_t = env.action_space.sample()
    a_t = noisy_action(a_t)
    s_tp1, r_t, done, info = env.step(np.array(a_t))
    q1, q2 = q_forward.apply(q_params, s_t, a_t)
    print(q1, q2)
    r_sum += r_t
    if done:
        avg_r.append(r_sum)
        if len(avg_r) >= 10:
            print(sum(avg_r)/10)
        r_sum = 0
        s_t = env.reset()

    replay_buffer.append([s_t, a_t, r_t, s_tp1, done]) 
    s_t = s_tp1   