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
                dict[(x, y)][a] = 0.0

    return dict

# 兩個都正面或背面，agent1的reward = 1、agent2的reward = -1；一正一負，agent2的reward = 1、agent1的reward = -1
def reward(states1, states2):
    if states1 == states2:
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
def nextAction(states, actions, qTables, trial, numOfEps=40,trials=1):

    #計算 epsilon-greedy 策略中的 epsilon 值
    if trial is not None and numOfEps > 0:
        epsilons = np.linspace(0.8, 0.1, numOfEps)
        index = int(trial // (trials / numOfEps))
        eps = epsilons[index]
    else:
        eps = 0.1

    numberOfAgents = len(qTables)
    new_actions = [0] * numberOfAgents
    for q in range(len(qTables)):
        if actions is not None and np.random.uniform() < eps:
            action = np.random.choice(actions)
        else:
            action = getKey(qTables[q][states], max(qTables[q][states].values()))
        new_actions[q] = action
    return new_actions
