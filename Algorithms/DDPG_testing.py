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
    a_pi = jax.vmap(partial(pi, pi_params))(s_t)
    return -jax.vmap(partial(q_forward, q_params))(s_t, a_pi).mean()

@jax.jit
def q_loss_fn(Q_s, Q_sp1, a_t, r_t, done):
        y = r_t + 0.99 * Q_sp1 * (1 - done)
        return (Q_s[a_t] - y)

@jax.jit
def critic_loss(q_params, pi_params, s_t, a_t, r_t, s_tp1, done):
    
    Q_s = q_forward(q_params, s_t, a_t)
    Q_sp1 = stop_gradient(q_forward(q_params, s_tp1, pi(pi_params, s_tp1)))

    losses = jax.vmap(q_loss_fn)(Q_s, Q_sp1, a_t, r_t, done)

    return 0.5 * jnp.square(losses).mean()

@jax.jit
def update(q_params, pi_params, optim_states, batch):
    s_t = jnp.array(batch.s, dtype=jnp.float32)
    a_t = jnp.array(batch.a, dtype=jnp.int32)
    r_t = jnp.array(batch.r, dtype=jnp.float32)
    s_tp1 = jnp.array(batch.s_p, dtype=jnp.float32)
    done = jnp.array(batch.d, dtype=jnp.float32)

    (q_optim_state, pi_optim_state) = optim_states

    q_loss, q_grads = jax.value_and_grad(critic_loss)(q_params, pi_params, s_t, a_t, r_t, s_tp1, done)
    updates, q_optim_state = q_optimizer.update(q_grads, q_optim_state, q_params)
    q_params = optax.apply_updates(q_params, updates)

    pi_loss, pi_grads = jax.value_and_grad(actor_loss)(q_params, pi_params, s_t)
    updates, pi_optim_state = pi_optimizer.update(pi_grads, pi_optim_state, pi_params)
    pi_params = optax.apply_updates(pi_params, updates)

    optim_states = (q_optim_state, pi_optim_state)
    loss = (q_loss, pi_loss)

    return loss, q_params, q_optim_state


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
        hk.Linear(1)
    ])
    return total_seq(s_seq(S)+a_seq(A))

# experience replay:
replay_buffer = deque(maxlen=1000000)
env = gym.make('Pendulum-v1')

rng = jax.random.PRNGKey(42)
q_params = q_val.init(rng, jnp.ones(env.observation_space.shape[0]), jnp.ones(env.action_space.shape[0]))
q_forward = hk.without_apply_rng(q_val).apply

pi_params = policy.init(rng, jnp.ones(env.observation_space.shape[0]))
pi = hk.without_apply_rng(policy).apply

q_optimizer = optax.chain(optax.adam(learning_rate=0.001))
q_optim_state = q_optimizer.init(q_params)

pi_optimizer = optax.chain(optax.adam(learning_rate=0.0001))
pi_optim_state = pi_optimizer.init(pi_params)


optim_states = (q_optim_state, pi_optim_state)

s_t = env.reset()
G = []

for i in range(1, TRAIN_STEPS): #https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html

    a_t = pi(pi_params, s_t)

    s_tp1, r_t, done, info = env.step(a_t)    

    replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

    #print('reward', r_t)

    if len(replay_buffer) > 128:
        batch = Transition(*zip(*random.sample(replay_buffer, k=128)))
        loss, params, optim_states = update(q_params, pi_params, optim_states, batch)

        if i % 100 == 0:
            print(i, ':', loss)

    s_t = s_tp1

    if done:
        #G.append(int(info['episode']['r']))
        #print(info)
        s_t = env.reset()

    """if i % VERBOSE_UPDATE == 0:
        avg_G = sum(G[-10:])/10
        print("Timestep: {}, Average return: {}".format(i, avg_G))"""

env.close()
