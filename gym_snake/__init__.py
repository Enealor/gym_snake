from gym.envs.registration import register
from gym_snake.wrappers import snake_wrapper
from gym_snake.utils import ascii_snake

register(
    id='snake-v0',
    entry_point='gym_snake.envs:SnakeEnv'
    shape=(20,20),
)

__all__ = ['snake_wrapper','SnakeEnv','ascii_snake']
