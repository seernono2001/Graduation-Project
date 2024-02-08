import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from functions import *
import pickle as pkl
import random

alpha = 0.0001
beta = 0.01
gamma = 0.9
trials = 1
# total actions
actions = [0, 1]

df = 0.8  # discount Factor
d_win = 0.0001
d_lose = 0.0002


def trainWolf():
    # Count的初始化
    C1 = create_Ctable()
    C2 = create_Ctable()
    # Policy Table的初始化x
    P1 = create_Policytable()
    P2 = create_Policytable()
    # Mean Policy Table的初始化
    MP1 = create_Policytable()
    MP2 = create_Policytable()
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
    head_agent1 = 0
    head_agent2 = 0
    states = (state_1, state_2)


    for i in range(20000):

        print("Iteration:" + str(i))
        
        # step(a)
        new_actions_agent1 = actions_select(states, P1, actions)
        new_actions_agent2 = actions_select(states, P2, actions)

        if(new_actions_agent1 == 1):
            head_agent1 += 1
        if(new_actions_agent2 == 1):
            head_agent2 += 1

        # step(b)
        new_state_1 = nextState(new_actions_agent1)
        new_state_2 = nextState(new_actions_agent2)
        new_states = (new_state_1, new_state_2)

        thisReward1, thisReward2 = reward(states)

        rewardSum1 = rewardSum1 + thisReward1
        rewardSum2 = rewardSum2 + thisReward2

        # 更新qTables
        qTables[0] = updateQ(qTables[0], states, new_states, new_actions_agent1, thisReward1, alpha, gamma)
        qTables[1] = updateQ(qTables[1], states, new_states, new_actions_agent2, thisReward2, alpha, gamma)

        # step 2(c)
        C1[states] = C1[states] + 1
        C2[states] = C2[states] + 1
        MP1 = update_meanpi(states, C1, MP1, P1, actions)
        MP2 = update_meanpi(states, C2, MP2, P2, actions)

        # step 2(d)
        P1 = update_pi(states, P1, MP1, qTables[0], d_win, d_lose, actions)
        P2 = update_pi(states, P2, MP2, qTables[1], d_win, d_lose, actions)

        # 下一回合的更新
        states = new_states

        ratio_1.append(head_agent1/(i+1))
        ratio_2.append(head_agent2/(i+1))
        iterationRewards1.append(rewardSum1)
        iterationRewards2.append(rewardSum2)

    plt.plot(list(range(20000)), ratio_1, "-", color="black")
    plt.title("Wolf")
    plt.xlabel("Iterations")
    plt.ylabel("Ratio(Head)")
    plt.savefig("./plots/Wolf_agent1_ratio.png")
    plt.clf()

    plt.plot(list(range(20000)), ratio_2, "-", color="black")
    plt.title("Wolf")
    plt.xlabel("Iterations")
    plt.ylabel("Ratio(Head)")
    plt.savefig("./plots/Wolf_agent2_ratio.png")
    plt.clf()

    plt.plot(list(range(20000)), iterationRewards1, "-", color="black")
    plt.title("Wolf")
    plt.xlabel("Iterations")
    plt.ylabel("Total Reward")
    plt.savefig("./plots/Wolf_agent1.png")
    plt.clf()

    plt.plot(list(range(20000)), iterationRewards2, "-", color="black")
    plt.title("Wolf")
    plt.xlabel("Iterations")
    plt.ylabel("Total Reward")
    plt.savefig("./plots/Wolf_agent2.png")
    plt.clf()

    pkl.dump(qTables[0], open("tables/q-table1.p", "wb"))
    pkl.dump(qTables[1], open("tables/q-table2.p", "wb"))

    P1_df = pd.DataFrame(P1)
    P2_df = pd.DataFrame(P2)

    # 將DataFrame轉換為Excel檔案
    excel_filename_P1 = 'P1_policy.xlsx'
    excel_filename_P2 = 'P2_policy.xlsx'
    P1_df.to_excel(excel_filename_P1, index=True)
    P2_df.to_excel(excel_filename_P2, index=True)


def main():
    trainWolf()


if __name__ == "__main__":
    main()