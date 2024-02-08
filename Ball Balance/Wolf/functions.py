import numpy as np
from helper import *


# Q-Table的初始化
def create_table():
    dict = {}
    discretePos = np.round(list(np.linspace(-1, 1, 100)), decimals=3)
    discreteV = np.round(list(np.linspace(-3, 3, 50)), decimals=3)
    actions = np.round(list(np.linspace(-1, 1, 15)), decimals=3)
   
    for p in discretePos:
        for v in discreteV:
            dict[(p, v)] = {}
            for a in actions:
                dict[(p, v)][a] = 0.0
    
    return dict

#Policy Table、Mean Policy Table的初始化
def create_Policytable():
    dict = {}
    discretePos = np.round(list(np.linspace(-1, 1, 100)), decimals=3)
    discreteV = np.round(list(np.linspace(-3, 3, 50)), decimals=3)
    actions = np.round(np.linspace(-1, 1, 15), decimals=3)
   
    for p in discretePos:
        for v in discreteV:
            dict[(p, v)] = {}
            for a in actions:
                dict[(p, v)][a] = (1.0/len(actions))
    
    return dict

#Count的初始化
def create_Ctable():
    dict = {}
    discretePos = np.round(list(np.linspace(-1, 1, 100)), decimals=3)
    discreteV = np.round(list(np.linspace(-3, 3, 50)), decimals=3)

    for p in discretePos:
        for v in discreteV:
            dict[(p, v)] = 0

    return dict
 

#論文的reward定義
def reward(x, v):
    R= 0.8 * np.exp(-(np.power(x, 2) / np.power(0.25, 2))) + 0.2 * np.exp(-(np.power(v, 2) / np.power(0.25, 2)))
    return R


#計算球的加速度(參數是論文提供的)
def dynamic(h1, h2, v):
    m = 0.5
    g = 9.8
    l = 2
    c = 0.01
    ball_acceleration = ((-c * v) + (m * g * ((h1 - h2) / l))) / m
    return ball_acceleration


#回傳球的位置(初始位置+移動距離(1/2at^2+v_0t))
def nextPosition(a, t, v, x_0):
    
    return 1 / 2 * a * t * t + v * t + x_0


#計算末速度
def nextSpeed(a, t, v_0):
    return a * t + v_0


#計算球的下一個states
def nextState(h1, h2, v, t, x_0, v_0):
    a = dynamic(h1, h2, v)
    nextX = nextPosition(a, t, v, x_0)
    nextS = nextSpeed(a, t, v_0)
    return nextX, nextS


# check if the new states are in discretized states
def isValidState(states):
    discretePos = np.round(list(np.linspace(-1, 1, 100)), decimals=3)
    position_space = 1 / 50
    position_index = np.abs(int(np.round(states[0] / position_space, decimals=0)))
    if states[0] > 0:
        position_index += 50
    else:
        position_index = 50 - position_index

    if position_index != 100 and position_index != 0:
        if position_index == 99:
            possible_positions = [discretePos[position_index - 1], discretePos[position_index]]
        else:
            possible_positions = [discretePos[position_index - 1], discretePos[position_index],
                                  discretePos[position_index + 1]]
    elif position_index == 100:
        possible_positions = [discretePos[position_index - 1], discretePos[position_index - 2]]
    else:
        possible_positions = [discretePos[position_index], discretePos[position_index + 1]]

    discreteV = np.round(list(np.linspace(-3, 3, 50)), decimals=3)
    velocity_space = 3 / 25
    velocity_index = np.abs(int(np.round(states[1] / velocity_space, decimals=0)))
    
    if states[1] > 0:
        velocity_index += 25
    else:
        velocity_index = 25 - velocity_index


    if velocity_index != 50 and velocity_index != 0:
        if velocity_index == 49:
            possible_velocities = [discreteV[velocity_index - 1], discreteV[velocity_index]]
        else:
            possible_velocities = [discreteV[velocity_index - 1], discreteV[velocity_index],
                                   discreteV[velocity_index + 1]]
    elif velocity_index == 50:
        possible_velocities = [discreteV[velocity_index - 1], discreteV[velocity_index - 2]]
    else:
        possible_velocities = [discreteV[velocity_index], discreteV[velocity_index + 1]]



    new_states = (min(possible_positions, key=lambda x: abs(x - states[0])),
                  min(possible_velocities, key=lambda x: abs(x - states[1])))
    
    return new_states


my_dict = create_table()


#下面開始是跟wolf有關的演算法

#Q-table的更新
def updateQ(qTable,state,next_state,action,R,alpha,gamma):

    maximum = 0 if not qTable[next_state] else max(qTable[next_state].values())
    
    qTable[state][action]=(1-alpha)*qTable[state][action]+alpha*(R+gamma*maximum)

    return qTable


#回傳agent的下一個action
def actions_select(state,Policy,actions,qTable):

    if actions is not None:
        p = []
        for a in actions:
            p.append(Policy[state][a])
    #print(p)
        return np.random.choice(actions, p = p)
    else:
        return getKey(qTable[state], max(qTable[state].values()))


#回傳delta
def delta(state,Q,Policy,MeanPolicy,d_win,d_lose,actions):
    sumPolicy = 0.0
    sumMeanPolicy = 0.0
    for i in actions:
        sumPolicy = sumPolicy+(Policy[state][i]*Q[state][i])
        sumMeanPolicy = sumMeanPolicy+(MeanPolicy[state][i]*Q[state][i])
    if (sumPolicy > sumMeanPolicy):
        return d_win
    else:
        return d_lose

def update_pi(state, Policy, MeanPolicy, Q, d_win, d_lose, actions):
    maxQValueIndex = getKey(Q[state], max(Q[state].values()))
    d_plus = delta(state, Q, Policy, MeanPolicy, d_win, d_lose, actions)
    d_minus = ((-1.0)*d_plus)/(len(actions) - 1.0)
    
    for i in actions:
        
        if (i == maxQValueIndex):
            Policy[state][i] = Policy[state][i] + d_plus
        else:
            Policy[state][i] = Policy[state][i] + d_minus

    values = np.array(list(Policy[state].values()))
    new = softmax(values)

    for i, action in enumerate(actions):
        Policy[state][action] = new[i]

    return Policy


#更新Mean Policy Table
def update_meanpi(state,C,MeanPolicy,Policy,actions):

    for i in actions:
        MeanPolicy[state][i] = MeanPolicy[state][i] + ((1.0/C[state]) * (Policy[state][i]-MeanPolicy[state][i]))
        
    values = np.array(list(MeanPolicy[state].values()))
    new = softmax(values)

    for i, action in enumerate(actions):
        MeanPolicy[state][action] = new[i]

    return	MeanPolicy

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)