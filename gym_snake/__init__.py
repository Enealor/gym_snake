from gym.envs.registration import register
from gym_snake.utils import ascii_snake
from gym_snake.envs import SnakeEnv

register(
    id='snake-v0',
    entry_point='gym_snake.envs:SnakeEnv',
    kwargs={'shape':(10,10)}
)

__all__ = ['SnakeEnv','ascii_snake']
