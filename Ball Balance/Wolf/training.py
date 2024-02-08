import pandas as pd
import matplotlib.pyplot as plt
from functions import *
import pickle as pkl
#np.set_printoptions(threshold=np.inf)

alpha = 0.9 #0.6
gamma = 0.9
samplingTime = 0.03
decimals = 3
trials = 5000  
actions = np.round(np.linspace(-1, 1, 15), decimals = decimals)

df = 0.8  #discount Factor
d_win = 0.3
d_lose = 0.6

def trainWolf():
    #step 1
    C = create_Ctable()
    P1 = create_Policytable()
    P2 = create_Policytable()
    MP1 = create_Policytable()
    MP2 = create_Policytable()
    qTable1 = create_table()
    qTable2 = create_table()
    qTables = [qTable1, qTable2]
    iterationRewards = []
    
    for i in range(5):
        rewardSumInTrial = []

        for trial in range(trials):
            progress(trial, trials, prefix = 'Iteration: ' + str(i))
            # Initialize states (x,xbar)
            states = (0.495, 1.041)
            rewardSum = 0
	    
            #step 2
            for t in np.arange(0, 20, samplingTime):

                #step 2(a)
                a1 = actions_select(states, P1, actions, None)
                a2 = actions_select(states, P2, actions, None)

                new_actions = [a1, a2]

                #執行完action的位置與速度
                x, v = nextState(h1 = new_actions[0], h2 = new_actions[1], v = states[1], t = samplingTime, x_0 = states[0],
                                     v_0 = states[1])
               
                # deal with velocities more than 3 and less than -3
                if v > 3: v = 3
                if v < -3: v = -3

                #球掉出桿子
                if np.abs(x) > 1 :
                    break

                #計算這次的reward
                thisReward = reward(states[0], states[1])

                # print(thisReward)

                #step 2(b)
                rewardSum = rewardSum + thisReward
                
                #將states四捨五入到小數第三位
                new_states = (np.round(x, decimals = decimals), np.round(v, decimals = decimals))

                # check if the new states are in discretized states
                new_states = isValidState(new_states)

                #更新qTable
                qTables[0] = updateQ(qTables[0], states, new_states, new_actions[0], thisReward, alpha, gamma)
                qTables[1] = updateQ(qTables[1], states, new_states, new_actions[1], thisReward, alpha, gamma)

                #step 2(c)
                C[states] = C[states] + 1
                MP1 = update_meanpi(states, C, MP1, P1, actions)                
                MP2 = update_meanpi(states, C, MP2, P2, actions)
                
                #step 2(d)
                P1 = update_pi(states, P1, MP1, qTables[0], d_win, d_lose, actions)
                P2 = update_pi(states, P2, MP2, qTables[1], d_win, d_lose, actions)

                    
                #更新states
                states = new_states

            #將這次trial的rewardSum存到一陣列
            rewardSumInTrial.append(rewardSum)
        
        #將這次iteration的rewardSum存到一陣列
        iterationRewards.append(rewardSumInTrial)

    # Create a DataFrame to store the rewards data
    reward_df = pd.DataFrame(iterationRewards)

    # Transpose the DataFrame to have trials as rows and iterations as columns
    reward_df = reward_df.transpose()

    # Write the DataFrame to an Excel file
    excel_filename = 'reward_data.xlsx'
    reward_df.to_excel(excel_filename, index=False)
    
    mean_output = np.mean(iterationRewards, axis=0)
    plt.plot(list(range(trials)), mean_output, '-', color="black")
    plt.title('Wolf')
    plt.xlabel("Trials")
    plt.ylabel("Average Total Reward")
    plt.savefig('./plots/Wolf.png')
    plt.clf()

    P1_df = pd.DataFrame(P1)
    P2_df = pd.DataFrame(P2)

    # 將DataFrame轉換為Excel檔案
    excel_filename_P1 = 'P1_policy.xlsx'
    excel_filename_P2 = 'P2_policy.xlsx'
    P1_df.to_excel(excel_filename_P1, index=True)
    P2_df.to_excel(excel_filename_P2, index=True)
    
    pkl.dump(qTables[0], open('tables/q-table1.p', 'wb'))
    pkl.dump(qTables[1], open('tables/q-table2.p', 'wb'))
    pkl.dump(P1, open('tables/policy-table1.p', 'wb'))
    pkl.dump(P2, open('tables/policy-table2.p', 'wb'))


def main():
    trainWolf()

if __name__ == '__main__':
    main()