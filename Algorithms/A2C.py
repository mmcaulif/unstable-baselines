import jax
import jax.nn
import jax.numpy as jnp
import numpy as onp
from jax.lax import stop_gradient
import haiku as hk
import optax
import gym
from gym.wrappers import RecordEpisodeStatistics
import random
import os
from functools import partial

# tell JAX to use CPU, cpu is faster on small networks
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

TRAIN_STEPS = 300000
VERBOSE_UPDATE = 1000

class A2C:
    def __init__(self,
                 learning_rate=0.0007,
                 epsilon=1,
                 value_coeff = 0.5,
                 entropy_coeff=0.001,
                 max_grad_norm=0.5): # Hyper parameters from stable baselines3 - https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
                 
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm

    def transform(self, rng, optimizer, model):
        self.optimizer = optimizer
        init_params = model.init(rng, jnp.ones(4))
        self.forward = hk.without_apply_rng(model).apply
        init_optim_state = self.optimizer.init(init_params)
        print("model initialised")
        return init_optim_state, init_params

    @partial(jax.jit, static_argnums = 0)
    def loss(self, params, s_t, a_t, r_t):
        val, policy = self.forward(params, s_t)
        adv = val - r_t
        c_loss = jnp.square(adv)
        action_probs = policy/jnp.sum(policy)
        log_probs = jnp.log(action_probs[a_t])
        entropy = -jnp.sum(action_probs * log_probs)
        a_loss = -jnp.log(action_probs) * stop_gradient(adv)

        loss = (self.value_coeff * c_loss) + a_loss + (entropy * self.entropy_coeff)
        print(loss)
        return loss

    #@partial(jax.jit, static_argnums = 0)
    def update(self, params, s_t, a_t, r_t, optim_state):
        loss, grads = jax.value_and_grad(self.loss)(params, s_t, a_t, r_t)
        updates, optim_state = self.optimizer.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, optim_state

    def act(self, policy):
        action_probs = onp.asanyarray(policy/jnp.sum(policy))
        action_list = jnp.arange(len(policy))
        a_t = onp.random.choice(action_list, p=action_probs)
        return a_t

@hk.transform 
def a2c_net(S):
    critic = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(1),
    ])

    actor = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(env.action_space.n), jax.nn.softmax
    ])

    return critic(S), actor(S)

# environment:
env = gym.make('CartPole-v0')
env = RecordEpisodeStatistics(env)

agent = A2C()

rng = jax.random.PRNGKey(42)
LR_SCHEDULE = optax.linear_schedule(agent.learning_rate, 0, TRAIN_STEPS, 0)
optimizer = optax.chain(optax.clip_by_global_norm(agent.max_grad_norm), optax.rmsprop(learning_rate=LR_SCHEDULE))
model = a2c_net
optim_state, params = agent.transform(rng, optimizer, model)

s_t = env.reset()
G = []

for i in range(1, TRAIN_STEPS):
    _ , policy = agent.forward(params, s_t)
    a_t = agent.act(policy)

    s_tp1, r_t, done, info = env.step(a_t)    

    loss, params, optim_state = agent.update(params, s_t, a_t, r_t, optim_state)

    s_t = s_tp1

    if done:
        G.append(int(info['episode']['r']))
        s_t = env.reset()

    if i % VERBOSE_UPDATE == 0:
        avg_G = sum(G[-10:])/10
        print("Timestep: {}, Average return: {}".format(i, avg_G))

env.close()