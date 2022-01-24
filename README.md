# Unstable-Baselines
'Optimised' implementations of popular Deep Reinforcement Algorithms in Jax, Optax and Haiku

## To do list:
* double dqn 
* make epsilon_greedy jit'able
* make a replay buffer class
* clean up dqn
* seperate environment and algo's
* make algo's environment agnostic
* make 'transform' into a seperate function that can be wrapped around an agent to initialise it
* make a baseline class
* create logging function
* create train on env function in baseline class
* clean up vmap usage in dqn. dont use two seperate loss functions, use: jax.vmap(partial(func, non vector parameters))(vector parameters) - need to look back at this
* focus on ddpg while struggling to debug a2c, maybe try implement multi-step
* get q-function portion of ddpg working then policy, then convert to td3
* test dqn on LunarLander
* clean up dqn code a bit more
* use stablebaselines3 and the environments to get targets in terms of learning for ddpg (pendulum-v1), a2c (1-step, cartpole) and dqn (in lunar lander)
* in the noise added in ddpg, change -env.action_space.low[0] to env.action_space.low[0] (no minus!)
