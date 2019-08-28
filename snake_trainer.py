# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:31:26 2019

@author: Eleanor
"""

import gym
from gym_snake import ascii_snake
from stable_baselines import DQN
from stable_baselines.deepq.policies import FeedForwardPolicy
import argparse
import tensorflow as tf
import numpy as np
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack

def snake_wrapper(env, stack_length=3):
    env = DummyVecEnv([lambda : env])
    env = VecFrameStack(env,stack_length)
    return env

parser = argparse.ArgumentParser(description='Have a CNN learn to play snake.')
parser.add_argument('-t','--time_steps', dest='time_steps', action='store',
                    default=10000, type=int,
                    help='The number of timesteps to iterate through when '+
                    'learning. Ignored during play behavior. (Default: 10000)')
parser.add_argument('-p','--play', dest='playing', action='store_true',
                    help='Watch the NN play. (Default: False)')
parser.add_argument('-e','--episodes', dest='episodes', action='store',
                    default=10, type=int,
                    help='The number of episodes to watch the snake play. An '+
                    'episode ends when the snake crashes into a wall or into '+
                    'their tail. (Default: 10)')
parser.add_argument('-f','--file_name', dest='file_name', action='store',
                    default='snake_model.pkl',type=str,
                    help='File name for the trained model for saving and '+
                    'loading. (Default: snake_model.pkl)')

def snake_cnn(images, **conv_kwargs):
    """
    CNN based on Nature paper. Made smaller as there is less visual
    information.
    """
    activ = tf.nn.relu
    h1 = activ(conv(images, 'c1', n_filters=16, filter_size=4, stride=1,
                    init_scale=np.sqrt(2), **conv_kwargs))
    h2 = activ(conv(h1, 'c2', n_filters=64, filter_size=2, stride=1,
                    init_scale=np.sqrt(2), **conv_kwargs))
    fl = conv_to_fc(h2)
    return activ(linear(fl, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))

class SnakePolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                     reuse=False, obs_phs=None, dueling=False, **_kwargs):
        super(SnakePolicy, self).__init__(sess, ob_space, ac_space, n_env,
                                          n_steps, n_batch, reuse,
                                          feature_extraction="cnn",
                                          obs_phs=obs_phs, dueling=dueling,
                                          cnn_extractor=snake_cnn,
                                          layer_norm=False, **_kwargs)

def play(episodes,file_name):
    env = gym.make("gym_snake:snake-v0")
    env = snake_wrapper(env)
    model = DQN.load(file_name,env)
    for _ in range(episodes):
        ascii_snake(env,model)

def learn(time_steps,file_name):
    env = gym.make('gym_snake:snake-v0')
    env = snake_wrapper(env)
    model = DQN(SnakePolicy,env,verbose=1)
    model.learn(time_steps)
    model.save(file_name)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.playing:
        play(args.episodes,args.file_name)
    else:
        learn(args.time_steps,args.file_name)
