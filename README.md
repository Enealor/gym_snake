# gym_snake
 An implementation of the game 'Snake' for the
 [OpenAI Gym](https://github.com/openai/gym) framework. Version 0.15.2 is
 required.

# The environment
 The Snake environment can be installed by running `pip install -e .` after
 downloading the setup.py and gym_snake folder. (This installs it in editable mode.) You can then create the game environment either directly, using
 ```
 from gym_snake import SnakeEnv

 env = SnakeEnv(**kwargs)
 ```
 or from Gym using
 ```
 import gym

 env = gym.make('gym_snake:snake-v0', **kwargs)
 ```
 where the arguments are
 ```
 shape : (int, int)
  This is the size of the grid

 length : int (optional)
  This is the starting length of the snake (defaults to 3).

 seed : int (optional)
  The seed is provided for so that runs can be reproduced.
  ```

# Getting started
 The script `snake_trainer.py` is provided as an example to both train and
 watch the snake after it has been trained. The snake is 'trained' using [DeepQ
 learning](https://www.nature.com/articles/nature14236) as implemented in
 [Baselines](https://github.com/openai/baselines). The snake cannot be
 watched while training, but you can call training it and then watching it with
 ```
 python snake_trainer.py
 ...
 python snake_trainer.py --play
 ```
 (There will be a lot of time between running the first script and it's
 completion.)
