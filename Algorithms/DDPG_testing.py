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
from gym.wrappers import RecordEpisodeStatistics
import random
import os
from functools import partial

# tell JAX to use CPU, cpu is faster on small networks
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

BUFFER_SIZE = 1000000
TRAIN_STEPS = 300000
TARGET_UPDATE = 10000
VERBOSE_UPDATE = 1000
EPSILON = 1

class Transition(NamedTuple):
    s: list  # state
    a: int  # action
    r: float  # reward
    s_p: list  # next state
    d: int  # done

@jax.jit
def actor_loss(q_params, pi_params, s_t):
    return -q_forward(q_params, s_t, pi(pi_params, s_t)).mean()

@jax.vmap
def noisy_action(a_t):
    noise = (jax.random.normal(rng, shape=a_t.shape) * 0.2).clip(-0.5, 0.5)
    return (a_t + noise).clip(-env.action_space.low[0],env.action_space.high[0])

@jax.jit
def q_loss_fn(Q_s, Q_sp1, r_t, done):
    y = r_t + 0.99 * Q_sp1 * (1 - done)
    return (Q_s - y)

@jax.jit
def critic_loss(q_params, pi_params, q_params_t, s_t, a_t, r_t, s_tp1, done):    
    Q_s = q_forward(q_params, s_t, a_t)
    
    a_pi = noisy_action(pi(pi_params, s_tp1))   #td3 style policy smoothing
    #a_pi = pi(pi_params, s_tp1)
    Q_sp1 = stop_gradient(q_forward(q_params_t, s_tp1, a_pi))

    losses = jax.vmap(q_loss_fn)(Q_s, Q_sp1, r_t, done)
    return 0.5 * jnp.square(losses).mean()

@jax.jit
def update(q_params, q_params_t, q_optim_state, batch):
    s_t = jnp.array(batch.s, dtype=jnp.float32)
    a_t = jnp.array(batch.a, dtype=jnp.int32)
    r_t = jnp.array(batch.r, dtype=jnp.float32)
    s_tp1 = jnp.array(batch.s_p, dtype=jnp.float32)
    done = jnp.array(batch.d, dtype=jnp.float32)

    q_loss, q_grads = jax.value_and_grad(critic_loss)(q_params, pi_params, q_params_t, s_t, a_t, r_t, s_tp1, done)
    updates, q_optim_state = q_optimizer.update(q_grads, q_optim_state, q_params)
    q_params = optax.apply_updates(q_params, updates)

    return q_loss, q_params, q_optim_state


@hk.transform
def policy(S):
    seq = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(env.action_space.shape[0]), jax.nn.tanh,
    ])
    return seq(S) * env.action_space.high[0]

@hk.transform
def q_val(S, A):
    s_seq = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
    ])
    a_seq = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
    ])
    total_seq = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(1),
    ])
    return total_seq(s_seq(S) + a_seq(A))

# experience replay:
replay_buffer = deque(maxlen=1000000)
env = gym.make('Pendulum-v1')

rng = jax.random.PRNGKey(42)

q_params = q_val.init(rng, jnp.ones(env.observation_space.shape[0]), jnp.ones(env.action_space.shape[0]))
q_params_t = hk.data_structures.to_immutable_dict(q_params)
q_forward = hk.without_apply_rng(q_val).apply

pi_params = policy.init(rng, jnp.ones(env.observation_space.shape[0]))
pi = hk.without_apply_rng(policy).apply

q_optimizer = optax.chain(optax.adam(learning_rate=0.001))
q_optim_state = q_optimizer.init(q_params)

polask_avg = lambda target, params: (1 - 0.005) * target + 0.005 * params

s_t = env.reset()
G = []
losses = []

for i in range(1, TRAIN_STEPS): #https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html

    a_t = pi(pi_params, s_t)

    s_tp1, r_t, done, info = env.step(a_t)    

    replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

    if i > 10000 and len(replay_buffer) > 128:
        batch = Transition(*zip(*random.sample(replay_buffer, k=128)))
        loss, q_params, q_optim_state = update(q_params, q_params_t, q_optim_state, batch)
        
        q_params_t = jax.tree_multimap(polask_avg, q_params_t, q_params)

        losses.append(loss)

        if i % 100 == 0:
            #print('Episodes:', i, 'critic loss:', loss)
            print('Episodes:', i, 'avg. critic loss:', sum(losses[-100:])/100)

    s_t = s_tp1

    if done:
        s_t = env.reset()

env.close()
