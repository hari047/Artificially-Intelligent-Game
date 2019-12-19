import numpy as np
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output
import random

model=load_model('AIAgame.h5')

def findLoc(state, obj):
    for i in range(0,8):
        for j in range(0,8):
            if (state[i,j] == obj).all():
                return i,j

def initGrid():
    state = np.zeros((8,8,4)) #4X4 is position of object. third dimension - each dimension for one object.
    #place player
    state[0,1] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    #place goal
    state[7,7] = np.array([1,0,0,0])
    return state

def getLoc(state, level):#
    for i in range(0,8):
        for j in range(0,8):
            if (state[i,j][level] == 1):
                return i,j

def getReward(state):
    player_loc = getLoc(state, 3)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)
    if (player_loc == pit):
        return -10
    elif (player_loc == goal):
        return 10
    else:
        return -1
    
def dispGrid(state):
    grid = np.zeros((8,8), dtype="str")
    player_loc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    for i in range(0,8):
        for j in range(0,8):
            grid[i,j] = ' '
            
    if player_loc:
        grid[player_loc] = 'P' #player
    if wall:
        grid[wall] = 'W' #wall
    if goal:
        grid[goal] = '+' #goal
    if pit:
        grid[pit] = '-' #pit
    
    return grid


state = initGrid()
grid = dispGrid(state)
    
testAlgo()

