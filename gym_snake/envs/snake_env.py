# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:03:49 2019

@author: Eleanor
"""
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
REPEAT = 4

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class Apple:
    def __init__(self, x,y):
        self.x, self.y = x,y

class Snake:
    def __init__(self, length,head_x=2,head_y=0,
                 initial_direction=RIGHT):
        # remember default length
        self._initial_length = length
        # set up snake length, position and direction
        self.length = length
        self.trail = None
        self.x = None
        self.y = None
        self.last_direction = [1,0]

        self.reset(head_x,head_y,initial_direction)

    def grow(self, length = 1):
        #Increases the size of the snake
        self.x.append(self.trail[0])
        self.y.append(self.trail[1])
        self.length += 1

    def slither(self,action):
        #Store trail for later growing purposes.
        self.trail = (self.x[-1],self.y[-1])
        #Update previous positions
        for i in range(self.length-1,0,-1):
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]

        # update position of head of snake
        xd,yd = self.direction(action)
        self.x[0] += xd
        self.y[0] += yd

    def reset(self,head_x,head_y,action):
        #Reset body length
        self.length = self._initial_length
        #The direction is opposite in construction so that it is as if the snake
        #was already heading in the direction of action.
        xd,yd = self.direction(action)
        self.x = [head_x-i*xd for i in range(self.length)]
        self.y = [head_y-i*yd for i in range(self.length)]

    def direction(self,action):
        #Takes the action and returns the direction of the snake
        if action == LEFT:
            direction_headed = [-1,0]
        elif action == RIGHT:
            direction_headed = [1,0]
        elif action == DOWN:
            direction_headed = [0,1]
        elif action == UP:
            direction_headed = [0,-1]
        else:
            direction_headed = self.last_direction
        self.last_direction = direction_headed
        return direction_headed

class SnakeEnv(gym.Env):

    metadata = {'render.modes': ['human','intensity','dict']}

    def __init__(self, shape, start_length=3, seed = None):
        super(SnakeEnv, self).__init__()
        self.seed(seed)
        self.nrow, self.ncol = shape
        self.reward_range = (-1,1)
        self.observation_space = spaces.Box(0,1,shape=[*shape,1])
        self.action_space = spaces.Discrete(5)
        self.score = 0

        self.snake = Snake(start_length)
        self.apple = Apple(0,0)
        #Reset the game board to ensure random start
        self.reset()

    def step(self, action):
        self.snake.slither(action)

        done = self.gameover()
        if done:
            reward = self.reward_range[0]
        elif self.apple_in_snake():
            self.ate_apple()
            reward = self.reward_range[1]
        else:
            reward = np.average(self.reward_range)

        out = self.render(mode='intensity')
        state_description = self.make_dict()

        return (out, reward, done, state_description)

    def gameover(self):
        return self.ate_tail() or self.outside_box()

    def ate_tail(self):
        '''
        Checks if the snake devoured their own tail. Returns True if they
        intersected their snake. Otherwise, returns False.
        '''
        head = (self.snake.x[0], self.snake.y[0])
        return head in zip(self.snake.x[1:],self.snake.y[1:])

    def outside_box(self):
        '''
        Checks if the snake is inside the grid constraints. If they are outside,
        returns True. Otherwise, returns False.
        '''
        x,y = (self.snake.x[0], self.snake.y[0])
        #easier to read as "not inside both constraints"
        return not (0<= x < self.ncol and 0 <= y < self.nrow)

    def apple_in_snake(self):
        return (self.apple.x,self.apple.y) in zip(self.snake.x,self.snake.y)

    def ate_apple(self):
        self.snake.grow()
        self.move_apple()
        self.score += 1

    def move_apple(self):
        self.apple.x = self.np_random.randint(self.ncol)
        self.apple.y = self.np_random.randint(self.nrow)
        while self.apple_in_snake():
            #If we are in the snake, try again
            self.apple.x = self.np_random.randint(self.ncol)
            self.apple.y = self.np_random.randint(self.nrow)

    def reset(self):
        snake_length = self.snake._initial_length
        head_x = self.np_random.randint(snake_length,self.ncol-snake_length)
        head_y = self.np_random.randint(snake_length,self.nrow-snake_length)
        orientation = self.np_random.randint(self.action_space.n)
        self.snake.reset(head_x,head_y,orientation)
        self.move_apple()
        self.score = 0
        return self.render('intensity')

    def render(self, mode='human'):
        if mode=='human':
            out = np.zeros((self.nrow,self.ncol), dtype = 'str')
            out[:] = ' '
            if not self.outside_box():
                out[(self.snake.y,self.snake.x)] = 'S'
            out[(self.apple.y,self.apple.x)] = 'A'
            print(out)
        elif mode=='intensity':
            out = np.zeros((self.nrow,self.ncol,1), dtype = 'float')
            if not self.outside_box():
                out[(self.snake.y,self.snake.x),0] = .5
            out[(self.apple.y,self.apple.x),0] = 1
            return out
        elif mode=='dict':
            return self.make_dict()

    def make_dict(self):
        return {'snake': list(zip(self.snake.x,self.snake.y)),
                'apple': (self.apple.x,self.apple.y),
                'score': self.score,
                'shape': (self.ncol,self.nrow)}

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
