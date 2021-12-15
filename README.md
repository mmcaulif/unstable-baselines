# Unstable-Baselines
'Optimised' implementations of popular Deep Reinforcement Algorithms in Jax, Optax and Haiku

## To do list:
* double dqn dqn
* make epsilon_greedy jit'able
* implement a vmap'able and jit'able huber loss function
* make a replay buffer class
* clean up dqn
* seperate environment and algo's
* make algo's environment agnostic
* make 'transform' into a seperate function that can be wrapped around an agent to initialise it
* make a baseline class
* create logging function
* create train on env function
* clean up vmap usage in dqn. dont use two seperate loss functions, use: jax.vmap(partial(func, non vector parameters))(vector parameters)
