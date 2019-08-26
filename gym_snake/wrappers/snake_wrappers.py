from gym.wrappers import FrameStack, TransformReward, TimeLimit
from gym.spaces import Box

class SnakeStack(FrameStack):
    def __init__(self,env,num_stack):
        super(SnakeStack,self).__init__(env,num_stack)
        shape = env.observation_space.shape
        self.observation_space = Box(0,1,shape=(*shape,num_stack))
        
def snake_wrapper(env,
               time_limit = 200, 
               default_reward=0,
               stack_length=3):
    env = TransformReward(env,reward_wrapper_func(default_reward))
    env = TimeLimit(env,time_limit)
    env = SnakeStack(env,stack_length)
    return env
    
def reward_wrapper_func(default_reward):
    def _reward_increase(reward):
        if reward == 0:
            return default_reward
        else:
            return reward
            
    return _reward_increase