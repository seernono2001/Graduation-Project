import numpy as np
from helper import *
import random

# Q-Table的初始化
def create_table():
    dict = {}

    X = 10
    Y = 10
    actions = ['UP','DOWN','RIGHT','LEFT','STAY']
   
    for x1 in range(X):
        for y1 in range(Y):
            for x2 in range(X):
                for y2 in range(Y):
                    for x3 in range(X):
                        for y3 in range(Y):
                            dict[(x1, y1, x2, y2, x3, y3)] = {}
                            for a in actions:
                                dict[(x1, y1, x2, y2, x3, y3)][a] = 0.0
    
    return dict

#Policy Table的初始化
def create_Ptable():
    dict = {}
   
    X = 10
    Y = 10
    actions = ['UP','DOWN','RIGHT','LEFT','STAY']

    for x1 in range(X):
        for y1 in range(Y):
            for x2 in range(X):
                for y2 in range(Y):
                    for x3 in range(X):
                        for y3 in range(Y):
                            dict[(x1, y1, x2, y2, x3, y3)] = {}
                            for a in actions:
                                dict[(x1, y1, x2, y2, x3, y3)][a] = 1/len(actions)
    
    return dict

#Count的初始化
def create_Ctable():
    dict = {}

    X = 10
    Y = 10
    actions = ['UP','DOWN','RIGHT','LEFT','STAY']

    for x1 in range(X):
        for y1 in range(Y):
            for x2 in range(X):
                for y2 in range(Y):
                    for x3 in range(X):
                        for y3 in range(Y):
                            dict[(x1, y1, x2, y2, x3, y3)] = {}
                            for a in actions:
                                dict[(x1, y1, x2, y2, x3, y3)][a] = 0

    return dict

    
#論文的reward定義
def reward(agent1, agent2, prey):
    if(agent1 == agent2):               #agent1和agent2重疊
        return (-10)
    elif(isAdjacent(agent1, prey) and isAdjacent(agent2, prey)):          #成功捕捉
        return (37.5)
    elif(isAdjacent(agent1, prey) and not isAdjacent(agent2, prey)):      #捕捉失敗(只有一個agent在prey的隔壁)
        return (-25)
    elif(isAdjacent(agent2, prey) and not isAdjacent(agent1, prey)):
        return (-25)
    else:                                                               
        return (-0.01)
    

#確認prey是否在agent的旁邊
def isAdjacent(agent,prey):
    if((prey[0]+1 == agent[0] and prey[1] == agent[1]) or (prey[0]-1 == agent[0] and prey[1] == agent[1]) or (prey[1]+1 == agent[1] and prey[0] == agent[0]) or (prey[1]-1 == agent[1] and prey[0] == agent[0])):
        return True
    else:
        return False


#根據action回傳下一個位置
def nextPos(pos,action):
    p = list(pos)

    if(action == "UP"):
        p[1] = pos[1]+1
    elif(action == "DOWN"):
        p[1] = pos[1]-1
    elif(action == "LEFT"):
        p[0] = pos[0]-1
    elif(action == "RIGHT"):
        p[0] = pos[0]+1
    elif(action == "STAY"):
        p = p
    else:
        print("UH-OH :(")

    if(p[0] < 0 or p[0] > 9 or p[1] < 0 or p[1] > 9):
        return pos

    return p


#回傳prey的下一個action
def nextAction(pos_agent1,pos_agent2,pos):
    rand_num = random.random()

    if rand_num <= 0.2:
        # The object stays on its current position
        return "STAY"
    else:
        # Get a list of available moves (adjacent empty cells)
        available_moves = prey_canMove(pos,pos_agent1,pos_agent2)
        if available_moves:
            # The object moves to one of the available empty cells with uniform probability
            return random.choice(available_moves)
        else:
            # There are no available moves, so the object stays in its current position
            return "STAY"


#確認prey可以進行的action
def prey_canMove(states, states_agent1, states_agent2):
    move = []
    can_UP = True
    can_LEFT = True
    can_RIGHT = True
    can_DOWN = True
    
    if(states[0]-1 == states_agent1[0] or states[0]-1 == states_agent2[0] or states[0]-1 < 0):
        can_LEFT=False
    if(states[0]+1 == states_agent1[0] or states[0]+1 == states_agent2[0] or states[0]+1 > 9):
        can_RIGHT=False
    if(states[1]-1 == states_agent1[1] or states[1]-1 == states_agent2[1] or states[1]-1 < 0):
        can_DOWN=False
    if(states[1]+1 == states_agent1[1] or states[1]+1 == states_agent2[1] or states[1]+1 > 9):
        can_UP=False
    
    if(can_LEFT==True):
        move.append("LEFT")
    if(can_RIGHT==True):
        move.append("RIGHT")
    if(can_DOWN==True):
        move.append("DOWN")
    if(can_UP==True):
        move.append("UP")

    return move


my_dict = create_table()


#Q-table的更新
def updateQ(qTable, state, next_state, action, R, alpha, gamma):

    maximum = 0 if not qTable[next_state] else max(qTable[next_state].values())

    qTable[state][action] = qTable[state][action] + alpha*(R+(gamma*maximum)-qTable[state][action])

    return qTable


#回傳agent的下一個action
def actions_select(payOff, state, actions, eps):

    p = np.random.random()

    if(p < eps):
        action = random.choice(actions)
    else:
        action = getKey(payOff[state], max(payOff[state].values()))

    return action

def updateP(p, states, action, lan, K):

    p[states][action] = p[states][action] + lan*(K - p[states][action])

    return p

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def updatePayoff(payOff, qTable, actions, states, max, min, w, pi):
    Q_max = max(qTable[states].values())
    Q_min = min(qTable[states].values())

    for action in actions:
        if(Q_max != Q_min):
            qTable[states][action] = (qTable[states][action] - Q_min)/(Q_max - Q_min)
        payOff[states][action] = (w * pi[states][action]) + ((1-w) * qTable[states][action])

    return payOff

def checkSuccess(r, d_win, d_lose):
    if(r == (-25) or r == (-10)):
        return 0, d_lose
    else:
        return 1, d_win