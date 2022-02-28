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

# tell JAX to use CPU, cpu is faster on small networks
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

BUFFER_SIZE = 1000000
TARGET_UPDATE = 10000
VERBOSE_UPDATE = 1000
EPSILON = 1
TAU = 0.005

class Transition(NamedTuple):
    s: list  # state
    a: int  # action
    r: float  # reward
    s_p: list  # next state
    d: int  # done

@jax.vmap
def noisy_action(a_t):
    noise = (jax.random.normal(rng, shape=a_t.shape) * 0.2).clip(-0.5, 0.5)
    return (a_t + noise).clip(-env.action_space.low[0],env.action_space.high[0])

@jax.jit
def q_loss_fn(Q_s, Q_sp1, r_t, done):
    y = r_t + 0.99 * Q_sp1 * (1 - done)
    return (Q_s - y)

@jax.jit
def critic_loss(q_params, q_params_t, pi_params_t, s_t, a_t, r_t, s_tp1, done):    
    Q_s1, Q_s2 = q_forward.apply(q_params, s_t, a_t)
    a_pi = noisy_action(pi_forward.apply(pi_params_t, s_tp1))
    Q1, Q2 = stop_gradient(q_forward.apply(q_params_t, s_tp1, a_pi))
    Q_sp1 = jnp.minimum(Q1, Q2)
    losses = jax.vmap(q_loss_fn)(Q_s1, Q_sp1, r_t, done) + jax.vmap(q_loss_fn)(Q_s2, Q_sp1, r_t, done)
    return 0.5 * jnp.square(losses).mean()

@jax.jit
def critic_update(q_params, q_params_t, pi_params_t, q_optim_state, batch):
    s_t = jnp.array(batch.s, dtype=jnp.float32)
    a_t = jnp.array(batch.a, dtype=jnp.int32)
    r_t = jnp.array(batch.r, dtype=jnp.float32)
    s_tp1 = jnp.array(batch.s_p, dtype=jnp.float32)
    done = jnp.array(batch.d, dtype=jnp.float32)    #move all this to a replay buffer class

    q_loss, q_grads = jax.value_and_grad(critic_loss)(q_params, q_params_t, pi_params_t, s_t, a_t, r_t, s_tp1, done)
    updates, q_optim_state = q_optimizer.update(q_grads, q_optim_state, q_params)
    q_params = optax.apply_updates(q_params, updates)

    return q_loss, q_params, q_optim_state

@jax.jit
def policy_loss(pi_params, q_params, s_t):
    a_pi = pi_forward.apply(pi_params, s_t)
    pi_loss, _ = jax.vmap(partial(q_forward.apply, q_params))(s_t, a_pi)
    return -jnp.mean(pi_loss)

@jax.jit
def policy_update(pi_params, q_params, pi_optim_state, batch):
    s_t = jnp.array(batch.s, dtype=jnp.float32)

    _, pi_grads = jax.value_and_grad(policy_loss)(pi_params, q_params, s_t)
    updates, pi_optim_state = pi_optimizer.update(pi_grads, pi_optim_state, pi_params)
    pi_params = optax.apply_updates(pi_params, updates)

    return pi_params, pi_optim_state

@hk.transform
def pi(S):
    seq = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(env.action_space.shape[0]), jax.nn.tanh,
    ])
    a_pi = seq(S) * env.action_space.high[0]
    return a_pi

@hk.transform
def q_val(S, A):
    SA = jnp.concatenate([S, A], axis=1)

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

# experience replay:
replay_buffer = deque(maxlen=100000)
env = gym.make('LunarLanderContinuous-v2')

rng = jax.random.PRNGKey(42)
critic_dims = (jnp.zeros((1,env.observation_space.shape[0])), jnp.zeros((1,env.action_space.shape[0])))
#print(*critic_dims)
q_params = q_val.init(rng, *critic_dims)
q_params_t = hk.data_structures.to_immutable_dict(q_params)
q_forward = hk.without_apply_rng(q_val)
q_optimizer = optax.adam(1e-3)
q_optim_state = q_optimizer.init(q_params)

pi_params = pi.init(rng, jnp.ones(env.observation_space.shape[0]))
pi_params_t = hk.data_structures.to_immutable_dict(pi_params)
pi_forward = hk.without_apply_rng(pi)
pi_optimizer = optax.adam(1e-4)
pi_optim_state = pi_optimizer.init(pi_params)

polask_avg = lambda target, params: (1 - TAU) * target + TAU * params

s_t = env.reset()
avg_r = deque(maxlen=10)
avg_loss = deque(maxlen=10)
r_sum = 0

for i in range(300000): #https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html
    a_t = pi_forward.apply(pi_params, s_t)
    a_t = noisy_action(a_t)
    s_tp1, r_t, done, info = env.step(np.array(a_t))
    r_sum += r_t
    if done:
        avg_r.append(r_sum)
        r_sum = 0
        s_t = env.reset()

    replay_buffer.append([s_t, a_t, r_t, s_tp1, done]) 
    s_t = s_tp1    

    if i >= 128:
        batch = Transition(*zip(*random.sample(replay_buffer, k=128)))
        q_loss, q_params, q_optim_state = critic_update(q_params, q_params_t, pi_params_t, q_optim_state, batch)            
        avg_loss.append(q_loss)

        if i % 2 == 0:  #td3 policy update delay
            pi_params, pi_optim_state = policy_update(pi_params, q_params, pi_optim_state, batch)
            q_params_t = jax.tree_multimap(polask_avg, q_params_t, q_params)
            pi_params_t = jax.tree_multimap(polask_avg, pi_params_t, pi_params)

    if i >= 500 and i % 100 == 0:
        print(f'Timesteps: {i} | avg. reward {sum(avg_r)/10} | avg. critic loss: {sum(avg_loss)/10}')   

env.close()
