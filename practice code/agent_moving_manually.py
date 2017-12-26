
import gym
import ppaquette_gym_super_mario
import random

import numpy as np
import gym 
from wrapper import action_space
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, Input, Lambda, merge
from keras.optimizers import RMSprop
from keras.models import Model
#from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np

from sum_tree import SumTree
import random


env = gym.make('ppaquette/meta-SuperMarioBros-v0')

wra_act=action_space

#reduce actions
env=wra_act.mario_action(env)

#reduce pixel


env=wra_act.ProcessFrame84(env)


#environment reset

env.reset()


action=[0,0,0,1,0,0]
action2=[0,0,0,1,1,0]
action3=[0,0,0,0,1,0]


for i in range(30):
    env.step(action)

for i in range(30
              ):
    env.step(action)

    env.step(action2)
    env.step(action3)
    env.step(action3)
    env.step(action2)
    env.step(action2)
    print(reward)
    


for i in range(20):
    env.step(action)

for i in range(10
              ):
    env.step(action)
    env.step(action2)
    env.step(action3)
    env.step(action3)
    env.step(action2)
    env.step(action2)

for i in range(7):
    env.step(action)


for i in range(25
              ):
    env.step(action)
    env.step(action2)
    env.step(action3)
    env.step(action3)
    env.step(action2)
    env.step(action2)

for i in range(2):
    env.step(action)

for i in range(30
              ):
    env.step(action)
    env.step(action2)
    env.step(action3)
    env.step(action3)
    env.step(action2)
    env.step(action2)

