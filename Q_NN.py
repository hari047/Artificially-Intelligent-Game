import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output
import random

#finds an array in the "depth" dimension of the grid
#gives the coordinates of all objects
def findLoc(state, obj):
    for i in range(0,8):
        for j in range(0,8):
            if (state[i,j] == obj).all():
                return i,j


#Initialize stationary grid, all items are placed deterministically
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

def makeMove(state, action):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    state = np.zeros((8,8,4))

    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    #[down,up,left,right]
    #e.g. up => (player row - 1, player column + 0)
    new_loc = (player_loc[0] + actions[action][0], player_loc[1] + actions[action][1])
    
    if (new_loc != wall):
        if ((np.array(new_loc) <= (7,7)).all() and (np.array(new_loc) >= (0,0)).all()):
            state[new_loc][3] = 1

    new_player_loc = findLoc(state, np.array([0,0,0,1]))
    if (not new_player_loc):
        state[player_loc] = np.array([0,0,0,1])
    #re-place pit
    state[pit][1] = 1
    #re-place wall
    state[wall][2] = 1
    #re-place goal
    state[goal][0] = 1

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

model = Sequential()
model.add(Dense(512, init='lecun_uniform', input_shape=(256,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(256, init='lecun_uniform'))#lucun_uniform
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(128, init='lecun_uniform'))#lucun_uniform
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

state = initGrid()
grid = dispGrid(state)

#op = model.predict(state.reshape(1,64), batch_size=1)

#Train the Algorithm
epochs = 10000
gamma = 0.9 
epsilon = 1
for i in range(epochs):
    
    state = initGrid()
    status = 1
    #while game still in progress
    while(status == 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,256), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,4)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state = makeMove(state, action)
        #Observe reward
        reward = getReward(new_state)
        #Get max_Q(S',a)
        newQ = model.predict(new_state.reshape(1,256), batch_size=1)
        maxQ = np.max(newQ)
        y = np.zeros((1,4))
        y[:] = qval[:]
        if reward == -1: #non-terminal state
            update = (reward + (gamma * maxQ))
        else: #terminal state
            update = reward
        y[0][action] = update #target output
        print("Game #: %s" % (i,))
        model.fit(state.reshape(1,256), y, batch_size=1, nb_epoch=1, verbose=1)
        state = new_state
        if reward != -1:
            if reward == 10:
                print('\n\n\n\n\n\n\n\n WON!! \n\n\n\n\n\n\n')
            status = 0

        clear_output(wait=True)
    if epsilon > 0.1:
        epsilon -= (1/epochs)
        
#Time to test our Learner
model.save('AIAgame.h5')
def testAlgo():
    i = 0
    state = initGrid()

    print("Initial State:")
    print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,256), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 30):
            print("Game lost; too many moves.")
            break
    
testAlgo()