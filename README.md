# gym_snake
 An implementation of the game 'Snake' for the OpenAI Gym framework
 (https://github.com/openai/gym). Requires version 0.15.2.

# Running
 The Snake environment can be installed by running 'pip install -e .' after
 downloading the setup.py and gym_snake folder. You can then create the game
 environment either directly, using
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
