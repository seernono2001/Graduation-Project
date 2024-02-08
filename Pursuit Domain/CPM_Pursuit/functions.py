import numpy as np
from helper import *
import random


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


#論文的reward
def reward(agent1,agent2,prey):
    if(agent1 == agent2):
        return (-10)
    elif(isAdjacent(agent1,prey) and isAdjacent(agent2,prey)):
        return (37.5)
    elif(isAdjacent(agent1,prey) and not isAdjacent(agent2,prey)):
        return (-25)
    elif(isAdjacent(agent2,prey) and not isAdjacent(agent1,prey)):
        return (-25)
    else:
        return (-0.01)

    
#檢查是否在隔壁
def isAdjacent(agent,prey):
    if((prey[0]+1 == agent[0] and prey[1] == agent[1]) or (prey[0]-1 == agent[0] and prey[1] == agent[1]) or (prey[1]+1 == agent[1] and prey[0] == agent[0]) or (prey[1]-1 == agent[1] and prey[0] == agent[0])):
        return True
    else:
        return False


#下一個位置
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

    if(p[0]<0 or p[0]>9 or p[1]<0 or p[1]>9):
        return pos

    return p


def nextAction(states, actions, payOffs, trial, numOfEps=40,trials=5000):

    #計算 epsilon-greedy 策略中的 epsilon 值
    if trial is not None and numOfEps > 0:
        epsilons = np.linspace(0.8, 0.1, numOfEps)
        index = int(trial // (trials / numOfEps))
        eps = epsilons[index]
    else:
        eps = 0.1

    numberOfAgents = len(payOffs)
    new_actions = [0] * numberOfAgents
    for q in range(len(payOffs)):
        if actions is not None and np.random.uniform() < eps:
            action = np.random.choice(actions)
        else:
            action = getKey(payOffs[q][states], max(payOffs[q][states].values()))
        new_actions[q] = action

    return new_actions


#獵物的下一個動作
def nextActPrey(pos_agent1,pos_agent2,pos):
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


#獵物可以執行的動作
def prey_canMove(states,states_agent1,states_agent2):
    move=[]
    can_UP=True
    can_LEFT=True
    can_RIGHT=True
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


def updatePayoff(payOff, qTable, pTable, states, actions, w):
    Qmax = max(qTable[states].values())
    Qmin = min(qTable[states].values())

    if(Qmax != Qmin):
        for act in actions:
            qTable[states][act] = (qTable[states][act] - Qmin) / (Qmax - Qmin)
    
    for action in actions:
        payOff[states][action] = (w * pTable[states][action]) + ((1 - w)*qTable[states][action])

    return payOff

def checkSuccess(r, d_lose, d_win):
    if(r == (-25) or r == (-10)):
        return 0, d_lose
    else:
        return 1, d_win

def updateP(pTable, states, actions, lam, K):
    for action in actions:
        pTable[states][action] += lam * (K - pTable[states][action])
    
    return pTable

def updateQ(qTable, states, next_states, action, alpha, r, gamma):
    maximum = 0 if not qTable[next_states] else max(qTable[next_states].values())
    
    qTable[states][action] = qTable[states][action] + alpha*(r + (gamma*maximum) - qTable[states][action])

    return qTable