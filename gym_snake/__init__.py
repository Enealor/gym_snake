from gym.envs.registration import register
from gym_snake.networks import snake_cnn, snake_cnn_big, snake_nn
from gym_snake.utils import enjoy_snake
from gym_snake.wrappers import snake_wrapper

register(
    id='snake-v0',
    entry_point='gym_snake.envs:SnakeEnv',
)

__all__ = ['enjoy_snake','snake_wrapper','SnakeEnv']