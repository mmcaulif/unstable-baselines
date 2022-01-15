import jax
import jax.nn
import jax.numpy as jnp
import numpy as onp
from jax.lax import stop_gradient
import haiku as hk
import optax
import gym
from gym.wrappers import RecordEpisodeStatistics
import os
from functools import partial
from collections import deque

# tell JAX to use CPU, cpu is faster on small networks
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')


#https://julien-vitay.net/deeprl/ActorCritic.html

TRAIN_STEPS = 300000
VERBOSE_UPDATE = 1000

class A2C:
    def __init__(self,
                 learning_rate=0.0007,
                 value_coeff = 0.5,
                 entropy_coeff=0.001,
                 gamma=0.99,
                 max_grad_norm=0.5,
                 n_steps=5): # Hyper parameters from stable baselines3 - https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
                 
        self.learning_rate = learning_rate
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps

    def transform(self, rng, optimizer, model):
        self.optimizer = optimizer
        init_params = model.init(rng, jnp.ones(4))
        self.forward = hk.without_apply_rng(model).apply
        init_optim_state = self.optimizer.init(init_params)
        print("model initialised")
        return init_optim_state, init_params

    @partial(jax.jit, static_argnums = 0)
    def actor_loss(self, params, s_t, a_t, r_t):
        val, policy = self.forward(params, s_t)
        action_probs = policy/jnp.sum(policy)
        log_prob = jnp.log(action_probs)
        policy_loss = -log_prob[a_t] * stop_gradient(val - r_t)
        entropy = -jnp.sum(action_probs * log_prob)
        return policy_loss + (entropy.mean() * self.entropy_coeff)

    @partial(jax.jit, static_argnums = 0)
    def critic_loss(self, params, s_t, r_t):
        val, _ = self.forward(params, s_t)
        adv = val - r_t
        return (0.5 * jnp.square(adv)) * self.value_coeff

    @partial(jax.jit, static_argnums = 0)
    def loss(self, params, s_t, a_t, r_t):
        a_loss = self.actor_loss(params, s_t, a_t, r_t)
        c_loss = self.critic_loss(params, s_t, r_t)
        loss = (c_loss + a_loss)[0]
        return loss

    @partial(jax.jit, static_argnums = 0)
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

    def get_n_step_value(self, trajectory):
        n_step_v = 0
        for i, v in enumerate(trajectory):
            n_step_v += v * (self.gamma ** i) 
        return n_step_v


@hk.transform 
def a2c_net(S):
    critic = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(1),
    ])

    actor = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
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

r_buffer = deque(maxlen=agent.n_steps)

s_t = env.reset()
episodes = 0
G = []

for i in range(1, TRAIN_STEPS):
    _ , policy = agent.forward(params, s_t)
    a_t = agent.act(policy)

    s_tp1, r_t, done, info = env.step(a_t)    

    r_buffer.append(r_t)

    print(r_buffer)
    print(agent.get_n_step_value(r_buffer))

    loss, params, optim_state = agent.update(params, s_t, a_t, r_t, optim_state)

    s_t = s_tp1

    if done:
        r_buffer.clear()
        G.append(int(info['episode']['r']))
        episodes += 1
        #print('Episode', E, 'done!')
        s_t = env.reset()

    if i % VERBOSE_UPDATE == 0:
        avg_G = sum(G[-10:])/10
        print("Timestep: {}, Average return: {}".format(i, avg_G))

env.close()