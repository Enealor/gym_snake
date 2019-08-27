# gym_snake
 An implementation of the game 'Snake' for the
 [OpenAI Gym](https://github.com/openai/gym) framework. This is done to make it
 easily connect to
 [stable-Baselines](https://github.com/hill-a/stable-baselines).

# The environment
 The Snake environment can be installed by running 'pip install .' after
 downloading the setup.py and gym_snake folder. You can then create the game
 from Gym using
 ```
 import gym

 env = gym.make('gym_snake:snake-v0')
 ```
 or using
 ```
 from gym_snake import SnakeEnv

 env = SnakeEnv(**kwargs)
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
 learning](https://www.nature.com/articles/nature14236). The snake
 cannot be watched while training, but you can tell it to train using `python
 snake_trainer.py` and then tell it to play using `python snake_trainer.py
 --play`. Training (can be) computationally intensive.
