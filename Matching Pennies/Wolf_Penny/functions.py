import numpy as np
from helper import *
import random


# Q-Table的初始化
def create_table():
    dict = {}

    X = 2
    Y = 2
    actions = [0, 1]

    for x in range(X):
        for y in range(Y):
            dict[(x, y)] = {}
            for a in actions:
                dict[(x, y)][a] = 0

    return dict


# Policy Table的初始化
def create_Policytable():
    dict = {}

    X = 2
    Y = 2
    actions = [0, 1]

    for x in range(X):
        for y in range(Y):
            dict[(x, y)] = {}
            for a in actions:            
                dict[(x, y)][a] = 1.0 / len(actions)

    return dict

# Count的初始化
def create_Ctable():
    dict = {}

    X = 2
    Y = 2

    for x in range(X):
        for y in range(Y):
            dict[(x, y)] = 0

    return dict


# 兩個都正面或背面，agent1的reward = 1、agent2的reward = -1；一正一負，agent2的reward = 1、agent1的reward = -1
def reward(states):
    if states[0] == states[1]:
        reward1 = 1
        reward2 = -1
    else:
        reward1 = -1
        reward2 = 1

    return reward1, reward2


# 決定是正面或背面
def nextState(action):
    if action == 1:
        return 1
    else:
        return 0

# Q-table的更新
def updateQ(qTable, state, next_state, action, R, alpha, gamma):
    maximum = 0 if not qTable[next_state] else max(qTable[next_state].values())

    qTable[state][action] = ((1 - alpha) * qTable[state][action]) + (alpha * (R + gamma * maximum))

    return qTable


# 回傳agent的下一個action
def actions_select(state, Policy, actions):
    p1 = []
    for a in actions:
        p1.append(Policy[state][a])

    if np.sum(p1) == 1.0:
        return np.random.choice(actions, p=p1)
    else:
        p1 = np.array(p1)
        p1 /= p1.sum()
        return np.random.choice(actions, p=p1)


# 回傳delta
def delta(state, Q, Policy, MeanPolicy, d_win, d_lose, actions):
    sumPolicy = 0.0
    sumMeanPolicy = 0.0
    for i in actions:
        sumPolicy = sumPolicy + (Policy[state][i] * Q[state][i])
        sumMeanPolicy = sumMeanPolicy + (MeanPolicy[state][i] * Q[state][i])
    if sumPolicy > sumMeanPolicy:
        return d_win
    else:
        return d_lose


#更新Policy Table
def update_pi(states, Policy, MeanPolicy, Q, d_win, d_lose, actions):
    sum1 = 0
    sum2 = 0

    maxQValueIndex = getKey(Q[states], max(Q[states].values()))
    d_plus = delta(states, Q, Policy, MeanPolicy, d_win, d_lose, actions)
    d_minus = ((-1.0)*d_plus)/(len(actions) - 1.0)
    
    for i in actions:
        
        if (i == maxQValueIndex):
            Policy[states][i] = Policy[states][i] + d_plus
        else:
            Policy[states][i] = Policy[states][i] + d_minus

        sum1 += Policy[states][i]


    # 使用 Softmax 函數進行正規化
    values = np.array(list(Policy[states].values()))
    new = softmax(values)

    for i, action in enumerate(actions):
        Policy[states][action] = new[i]
        sum2 += Policy[states][action]

    # new = softmax(values)
    # for i, action in enumerate(actions):
    #     Policy[states][action] = new[i]
    #     sum += Policy[states][action]

    return Policy


#更新Mean Policy Table
def update_meanpi(states,C,MeanPolicy,Policy,actions):
    sum1 = 0
    sum2 = 0

    for i in actions:
        MeanPolicy[states][i] = MeanPolicy[states][i] + ((1.0/C[states]) * (Policy[states][i]-MeanPolicy[states][i]))
        sum1 += MeanPolicy[states][i]

    # values = np.array(list(MeanPolicy[states].values()))
    # new = softmax(values)
    # for i, action in enumerate(actions):
    #     MeanPolicy[states][action] = new[i]

    # 使用 Softmax 函数进行正规化
    values = np.array(list(MeanPolicy[states].values()))
    new = softmax(values)

    for i, action in enumerate(actions):
        MeanPolicy[states][action] = new[i]
        sum2 += MeanPolicy[states][action]

    return	MeanPolicy

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
