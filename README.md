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