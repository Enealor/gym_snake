# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:31:26 2019

@author: Eleanor
"""

import gym
from gym_snake import snake_wrapper as snake_skin, enjoy_snake
from gym_snake.networks import snake_cnn
from baselines import deepq
import argparse

parser = argparse.ArgumentParser(description='Have a CNN learn to play snake.')
parser.add_argument('-p','--play', dest='playing', action='store_true', 
                    help='Watch the NN play. (Default: learn instead)')
parser.add_argument('-s','--shape', dest='shape', action='store', nargs=2,
                    default=(10,10), type=int,
                    help='Size of game board. Must be the same when learning '+
                    'and playing (Default: 10 by 10)')
parser.add_argument('-e','--epochs', dest='epochs', action='store',
                    default=100000, type=int,
                    help='The number of timesteps to iterate through when '+
                    'learning. Ignored during play behavior. (Default: 100000)')
parser.add_argument('-f','--file_name', dest='file_name', action='store',
                    default='snake_model.pkl',type=str,
                    help='File name for the trained model for saving and '+
                    'loading.')

def play(shape):
    env = gym.make("gym_snake:snake-v0",shape=shape)
    env = snake_skin(env)
    model = deepq.learn(
        env,
        snake_cnn,
        total_timesteps=0,
        load_path="snake_model.pkl"
    )
    try:
        while True:
            enjoy_snake(env,model)
    except:
        pass

def learn(shape,epochs):
    env = gym.make('gym_snake:snake-v0',shape=shape)
    env = snake_skin(env,default_reward=0.01)
    act = deepq.learn(
        env,snake_cnn,
        lr=1e-3,
        total_timesteps=epochs,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        print_freq=100,
        prioritized_replay = True
    )
    print("Saving model to snake_model.pkl")
    act.save("snake_model.pkl")


if __name__ == '__main__':
    args = parser.parse_args()
    if args.playing:
        play(args.shape)
    else:
        learn(args.shape,args.epochs)