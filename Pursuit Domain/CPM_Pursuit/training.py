import matplotlib.pyplot as plt
from functions import *
import pickle as pkl
import random

alpha = 0.6
beta = 0.55
gamma = 0.9
trials = 30000
w = 0.3
d_win = 0.3
d_lose = 0.6


#total actions
actions = ['UP','DOWN','RIGHT','LEFT','STAY'] 

def trainCPM():
    # create q-Table
    qTable1 = create_table()    #agent1 q-table
    qTable2 = create_table()    #agent2 q-table
    qTables = [qTable1, qTable2]

    pTable1 = create_Ptable()
    pTable2 = create_Ptable()
    payOff1 = create_table()
    payOff2 = create_table()
    payOffs = [payOff1, payOff2]

    iterationCapture = []
    iterationSteps = []

    for i in range(20):
        captureInTrial=[]
        stepsInTrial=[]
        for trial in range(trials):
            progress(trial, trials, prefix='Iteration: ' + str(i))
            
            #agent1, agent2, prey初始位置
            pos_prey = [random.randint(0,9), random.randint(0,9)]
            pos_agent1 = [random.randint(0,9), random.randint(0,9)]
            while(pos_agent1 == pos_prey):
                pos_agent1 = [random.randint(0,9), random.randint(0,9)]
            pos_agent2 = [random.randint(0,9), random.randint(0,9)]
            while(pos_agent2 == pos_agent1 or pos_agent2 == pos_prey):
                pos_agent2 = [random.randint(0,9), random.randint(0,9)]

            #agent1, agent2, prey初始state
            states = (pos_agent1[0], pos_agent1[1], pos_agent2[0], pos_agent2[1], pos_prey[0], pos_prey[1])

            capture = 0                 #成功捕捉次數
            firstCaptured = False       #確認第一次捕捉的步數
            steps = 1000

            for t in np.arange(0, 1000):

                payOffs[0] = updatePayoff(payOffs[0], qTables[0], pTable1, states, actions, w)
                payOffs[1] = updatePayoff(payOffs[1], qTables[1], pTable2, states, actions, w)

                #決定agent1, agent2的action
                new_actions = nextAction(states, actions, payOffs, trial, numOfEps = 40, trials = trials)
                
                #決定prey的action
                action_prey = nextActPrey(pos_agent1, pos_agent2, pos_prey)

                ##agent1, agent2, prey移動後的位置
                new_pos_agent1 = nextPos(pos_agent1, new_actions[0])
                new_pos_agent2 = nextPos(pos_agent2, new_actions[1])
                new_pos_prey = nextPos(pos_prey, action_prey)

                #確認是否撞牆，撞牆reward給 -25.5
                if(new_pos_agent1 == pos_agent1 and new_actions[0] != "STAY"):
                    reward1 = (-25.5)
                else:
                    reward1 = reward(pos_agent1, pos_agent2, pos_prey)
                    
                if(new_pos_agent2 == pos_agent2 and new_actions[1] != "STAY"):
                    reward2 = (-25.5)
                else:
                    reward2 = reward(pos_agent1, pos_agent2, pos_prey)


                #更新states
                new_states = (new_pos_agent1[0], new_pos_agent1[1], new_pos_agent2[0], new_pos_agent2[1], new_pos_prey[0], new_pos_prey[1])
                
                K1, lam1 = checkSuccess(reward1, d_lose, d_win)
                K2, lam2 = checkSuccess(reward2, d_lose, d_win)

                pTable1 = updateP(pTable1, states, actions, lam1, K1)
                pTable2 = updateP(pTable2, states, actions, lam2, K2)

                #更新qTables
                qTables[0] = updateQ(qTables[0], states, new_states, new_actions[0], alpha, reward1, gamma)
                qTables[1] = updateQ(qTables[1], states, new_states, new_actions[1], alpha, reward2, gamma)

                #若重疊或捕捉失敗或捕捉成功，重新分配位置
                #重疊或捕捉失敗
                if(reward1 == (-10) or reward1 == (-25) or reward2 == (-10) or reward2 == (-25)):
                    new_pos_agent1 = [random.randint(0,9), random.randint(0,9)]
                    while(new_pos_agent1 == pos_prey):
                        new_pos_agent1 = [random.randint(0,9), random.randint(0,9)]
                    new_pos_agent2 = [random.randint(0,9), random.randint(0,9)]
                    while(new_pos_agent2 == new_pos_agent1 or new_pos_agent2 == pos_prey):
                        new_pos_agent2 = [random.randint(0,9), random.randint(0,9)]
                #捕捉成功
                if(reward1 == 37.5 or reward2 == 37.5):
                    new_pos_prey = [random.randint(0,9), random.randint(0,9)]
                    new_pos_agent1 = [random.randint(0,9), random.randint(0,9)]
                    while(new_pos_agent1 == new_pos_prey):
                        new_pos_agent1 = [random.randint(0,9), random.randint(0,9)]
                    new_pos_agent2 = [random.randint(0,9), random.randint(0,9)]
                    while(new_pos_agent2 == new_pos_agent1 or new_pos_agent2 == new_pos_prey):
                        new_pos_agent2 = [random.randint(0,9), random.randint(0,9)]
                    capture = capture + 1

                #如果有重新分配位置，再更新states一次
                new_states = (new_pos_agent1[0], new_pos_agent1[1], new_pos_agent2[0], new_pos_agent2[1], new_pos_prey[0], new_pos_prey[1])

                #下一回合的更新
                states = new_states
                pos_agent1 = new_pos_agent1
                pos_agent2 = new_pos_agent2
                pos_prey = new_pos_prey

                #記錄第一次捕捉成功的步數
                if(capture == 1 and firstCaptured == False):
                    steps = t
                    firstCaptured = True

            captureInTrial.append(capture)
            stepsInTrial.append(steps)
        
        iterationCapture.append(captureInTrial)
        iterationSteps.append(stepsInTrial)

    mean_capture = np.mean(iterationCapture, axis = 0)
    mean_steps = np.mean(iterationSteps, axis = 0)
    
    pkl.dump(mean_capture, open('tables/captures.p', 'wb'))
    pkl.dump(mean_steps, open('tables/steps.p', 'wb'))


def main():
    trainCPM()

if __name__ == '__main__':
    main()