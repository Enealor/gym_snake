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
 
class Snake:
    def __init__(self, length,initial_position=(2,0),
                 initial_direction=RIGHT):
        # remember default length
        self.initial_length = length
        # set up snake length, position and direction
        self.length = length
        self.trail = None
        self.x = None
        self.y = None
        self.last_direction = [1,0]
        
        self.reset(initial_position,initial_direction)
        
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
        xdir,ydir = self.action_to_direction(action)
        self.x[0] += xdir
        self.y[0] += ydir
        
    def reset(self,head_location,direction_headed):
        #Reset body length
        self.length = self.initial_length
        #Reset direction
        xdir,ydir = self.action_to_direction(direction_headed)
        #Reset body. The body is in the direction opposite of the direction 
        #headed.
        xi, yi = head_location
        self.x = [xi-i*xdir for i in range(self.length)]
        self.y = [yi-i*ydir for i in range(self.length)]
        
    def action_to_direction(self,action):
        #Processes action into a direction. If the action is not recognized,
        #then the direction does not change.
        if action == LEFT:
            self.direction = [-1,0]
        elif action == RIGHT:
            self.direction = [1,0]
        elif action == DOWN:
            self.direction = [0,-1]
        elif action == UP:
            self.direction = [0,1]
        else:
            pass
        return self.direction
    
class SnakeEnv(gym.Env):

    metadata = {'render.modes': ['human','intensity','dict']}

    def __init__(self, shape, start_length=3, seed = None):
        super(SnakeEnv, self).__init__()
        self.seed(seed)
        self.nrow, self.ncol = shape
        self.reward_range = (-1,1)
        
        self.snake = Snake(start_length)
        self.apple = (0,0)
        self.move_apple()
        
        self.observation_space = spaces.Box(0,1,shape=shape)
        self.action_space = spaces.Discrete(4)

    def step(self, action):
        self.snake.slither(action)
        
        done = self.gameover()
        if done:
            reward = self.reward_range[0]
        elif self.apple in zip(self.snake.x,self.snake.y):
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
    
    def ate_apple(self):
        self.snake.grow()
        self.move_apple()
        self.score += 1
    
    def move_apple(self):
        snake = list(zip(self.snake.x,self.snake.y))
        self.apple = (self.np_random.randint(self.ncol),
                      self.np_random.randint(self.nrow))
        while self.apple in snake:
            self.apple = (self.np_random.randint(self.ncol),
                          self.np_random.randint(self.nrow))

    def reset(self):
        length = self.snake.initial_length
        head_x = self.np_random.randint(length,self.ncol-length)
        head_y = self.np_random.randint(length,self.nrow-length)
        orientation = self.np_random.randint(self.action_space.n)
        self.snake.reset((head_x,head_y),orientation)
        self.move_apple()
        self.score = 0
        return self.render('intensity')
      
    def render(self, mode='human'):
        if mode=='human':
            out = np.zeros((self.nrow,self.ncol), dtype = 'str')
            out[:] = ' '
            if not self.outside_box():
                out[(self.snake.y,self.snake.x)] = 'S'
            out[self.apple] = 'A'
            print(out)
        elif mode=='intensity':
            out = np.zeros((self.nrow,self.ncol), dtype = 'float')
            if not self.outside_box():
                out[(self.snake.y,self.snake.x)] = .5
            out[self.apple] = 1
            return out
        elif mode=='dict':
            return self.make_dict()
            
    def make_dict(self):
        return {'snake': list(zip(self.snake.x,self.snake.y)),
                    'apple': self.apple,
                    'score': self.score,
                    'shape': (self.ncol,self.nrow)}

    def close(self):
        pass
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]