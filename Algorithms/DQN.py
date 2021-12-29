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
TRAIN_STEPS = 1500000
TARGET_UPDATE = 10000
VERBOSE_UPDATE = 1000
EPSILON = 1

class Transition(NamedTuple):
    s: list  # state
    a: int  # action
    r: float  # reward
    s_p: list  # next state
    d: int  # done

class DQN:
    def __init__(self,
                 learning_rate=0.0001,
                 buffer_size=1000000,
                 learning_starts=50000,
                 batch_size=32,
                 gamma=0.99,
                 train_freq=4,
                 target_update_interval=10000,
                 epsilon=1,
                 max_grad_norm=10): # Hyper parameters from stable baselines3 - https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
                 
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm

    def transform(self, rng, optimizer, model, obs_dims):
        self.optimizer = optimizer
        init_params = model.init(rng, obs_dims)
        self.forward = hk.without_apply_rng(model).apply
        init_optim_state = self.optimizer.init(init_params)
        target_params = hk.data_structures.to_immutable_dict(init_params)
        print("model initialised")
        return init_optim_state, init_params, target_params

    def q_loss_fn(self, Q_s, Q_sp1, a_t, r_t, done):
        Q_target = r_t + self.gamma * Q_sp1.max() * (1 - done)
        return (Q_s[a_t] - Q_target)

    @partial(jax.jit, static_argnums = 0)
    def mse_loss(self, params, target_params, s_t, a_t, r_t, s_tp1, done):
        Q_s = self.forward(params, s_t)
        Q_sp1 = stop_gradient(self.forward(target_params, s_tp1))

        losses = jax.vmap(self.q_loss_fn)(Q_s, Q_sp1, a_t, r_t, done)

        return 0.5 * jnp.square(losses).mean()

    @partial(jax.jit, static_argnums = 0)
    def update(self, params, target_params, optim_state, batch):
        s_t = jnp.array(batch.s, dtype=jnp.float32)
        a_t = jnp.array(batch.a, dtype=jnp.int32)
        r_t = jnp.array(batch.r, dtype=jnp.float32)
        s_tp1 = jnp.array(batch.s_p, dtype=jnp.float32)
        done = jnp.array(batch.d, dtype=jnp.float32)

        loss, grads = jax.value_and_grad(self.mse_loss)(params, target_params, s_t, a_t, r_t, s_tp1, done)
        updates, optim_state = self.optimizer.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)

        return loss, params, optim_state

    # need to make anneal from a max to a min instead of current method
    def epsilon_greedy(self, s_t, params):
        rand = random.random()
        if rand < self.epsilon:
            a_t = env.action_space.sample()
        else:
            a_t = int(jnp.argmax(self.forward(params, s_t)))
        self.epsilon = self.epsilon * 0.99
        return a_t

# initialisations
@hk.transform  # stable baselines3 dqn network is input_dim, 64, 64, output_dim
def vanilla_net(S):
    seq = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(env.action_space.n),  # , w_init=jnp.zeros
    ])
    return seq(S)

@hk.transform 
def dueling_net(S):
    val = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(1),
    ])

    adv = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(env.action_space.n),
    ])

    return val(S) + adv(S)

# environment:
env = gym.make('LunarLander-v2')   #gym.make('CartPole-v0')
env = RecordEpisodeStatistics(env)

# agent
dqn_agent = DQN()

# experience replay:
replay_buffer = deque(maxlen=dqn_agent.buffer_size)

# initialisation
rng = jax.random.PRNGKey(42)
LR_SCHEDULE = optax.linear_schedule(dqn_agent.learning_rate, 0, TRAIN_STEPS, dqn_agent.learning_starts)
optimizer = optax.chain(optax.clip_by_global_norm(dqn_agent.max_grad_norm), optax.adam(learning_rate=LR_SCHEDULE))
model = dueling_net
dimensions = jnp.ones(env.observation_space.shape[0])
optim_state, params, target_params = dqn_agent.transform(rng, optimizer, model, dimensions)

s_t = env.reset()
E = 0
G = []

for i in range(1, TRAIN_STEPS):
    a_t = dqn_agent.epsilon_greedy(jnp.asarray(s_t), params)

    s_tp1, r_t, done, info = env.step(a_t)    

    replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

    if i % dqn_agent.train_freq == 0 and i > dqn_agent.learning_starts and len(replay_buffer) > dqn_agent.batch_size:
        batch = Transition(*zip(*random.sample(replay_buffer, k=dqn_agent.batch_size)))
        loss, params, optim_state = dqn_agent.update(params, target_params, optim_state, batch)

    s_t = s_tp1

    if i % dqn_agent.target_update_interval == 0:
            target_params = hk.data_structures.to_immutable_dict(params)

    if done:
        E += 1  #should solve lunar lander in ~1200 episodes
        print('Episode: ', E, 'done!')
        G.append(int(info['episode']['r']))
        s_t = env.reset()

    if i % VERBOSE_UPDATE == 0:
        avg_G = sum(G[-10:])/10
        print("Timestep: {}, Average return: {}".format(i, avg_G))

env.close()
