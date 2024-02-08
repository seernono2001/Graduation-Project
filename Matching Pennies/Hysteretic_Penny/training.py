import pandas as pd
import matplotlib.pyplot as plt
from functions import *
import pickle as pkl
import random
from hysteretic import *

alpha = 0.1
beta = 0.01
gamma = 0.9
# total actions
actions = [0, 1]


def trainHysteretic():
    # create q-Table
    qTable1 = create_table()  # agent1 q-table
    qTable2 = create_table()  # agent2 q-table
    qTables = [qTable1, qTable2]

    iterationRewards1 = []
    iterationRewards2 = []
    ratio_1 = []
    ratio_2 = []
    state_1 = random.randint(0, 1)
    state_2 = random.randint(0, 1)
    rewardSum1 = 0
    rewardSum2 = 0
    states = (state_1, state_2)
    trial = 1
    head_agent1 = 0
    head_agent2 = 0

    for i in range(1000000):

        print("Iteration:" + str(i))
        
        new_actions = nextAction(states, actions, qTables, trial, numOfEps=40, trials=1)

        if(new_actions[0] == 1):
            head_agent1 += 1
        if(new_actions[1] == 1):
            head_agent2 += 1

        new_state_1 = nextState(new_actions[0])
        new_state_2 = nextState(new_actions[1])
        new_states = (new_state_1, new_state_2)

        thisReward1, thisReward2 = reward(state_1, state_2)

        rewardSum1 = rewardSum1 + thisReward1
        rewardSum2 = rewardSum2 + thisReward2

        # 更新qTables
        qTables[0] = hysteretic(qTables[0], states, new_actions[0], alpha, beta, thisReward1, gamma, new_states)
        qTables[1] = hysteretic(qTables[1], states, new_actions[1], alpha, beta, thisReward2, gamma, new_states)

        # 下一回合的更新
        state_1 = new_state_1
        state_2 = new_state_2
        states = (state_1, state_2)

        ratio_1.append(head_agent1/(i+1))
        ratio_2.append(head_agent2/(i+1))
        iterationRewards1.append(rewardSum1)
        iterationRewards2.append(rewardSum2)

    
    plt.plot(list(range(1000000)), ratio_1, "-", color="black")
    plt.title("Hysteretic")
    plt.xlabel("Iterations")
    plt.ylabel("Ratio(Head)")
    plt.savefig("./plots/Hysteretic_agent1_ratio.png")
    plt.clf()

    plt.plot(list(range(1000000)), ratio_2, "-", color="black")
    plt.title("Hysteretic")
    plt.xlabel("Iterations")
    plt.ylabel("Ratio(Head)")
    plt.savefig("./plots/Hysteretic_agent2_ratio.png")
    plt.clf()

    plt.plot(list(range(1000000)), iterationRewards1, "-", color="black")
    plt.title("Hysteretic")
    plt.xlabel("Iterations")
    plt.ylabel("Total Reward")
    plt.savefig("./plots/Hysteretic_agent1.png")
    plt.clf()

    plt.plot(list(range(1000000)), iterationRewards2, "-", color="black")
    plt.title("Hysteretic")
    plt.xlabel("Iterations")
    plt.ylabel("Total Reward")
    plt.savefig("./plots/Hysteretic_agent2.png")
    plt.clf()

    pkl.dump(qTables[0], open("tables/q-table1.p", "wb"))
    pkl.dump(qTables[1], open("tables/q-table2.p", "wb"))


def main():
    trainHysteretic()


if __name__ == "__main__":
    main()