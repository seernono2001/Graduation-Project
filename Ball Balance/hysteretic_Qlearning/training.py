from hysteretic import *
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
import pickle as pkl

alpha = 0.9
beta = 0.1
gamma = 0.9
samplingTime = 0.03
decimals = 3
trials = 5000
#total actions
actions = np.round(np.linspace(-1, 1, 15), decimals=decimals)   



def trainHysteretic():
    # create q-Table
    qTable1 = create_table()    #agent1 q-table
    qTable2 = create_table()    #agent2 q-table
    qTables = [qTable1, qTable2]

    iterationRewards = []
    for i in range(5):
        rewardSumInTrial = []
        for trial in range(trials):
            progress(trial, trials, prefix='Iteration: ' + str(i))
            # Initialize states (x,xbar)
            states = (0.495, 1.041)  
            rewardSum = 0

            #Each trial  goes on at the most 20 seconds with samplingTime = 0.03
            for t in np.arange(0, 20, samplingTime):

                new_actions = nextAction(states, actions, qTables, trial, numOfEps=40, trials=trials)
                
                # print(new_actions)

                x, v = nextState(h1=new_actions[0], h2=new_actions[1], v=states[1], t=samplingTime, x_0=states[0],
                                     v_0=states[1])
                
                # print(x)
                # print(v)

                # deal with velocities more than 3 and less than -3
                if v > 3: v = 3
                if v < -3: v = -3

                if np.abs(x) > 1:
                    break

                thisReward = reward(states[0], states[1])
                
                #print(thisReward)

                rewardSum = rewardSum + thisReward

                #print(rewardSum)

                new_states = (np.round(x, decimals=decimals), np.round(v, decimals=decimals))

                # print("Before")
                # print(new_states)

                # check if the new states are in discretized states
                new_states = isValidState(new_states)

                # print("After")
                # print(new_states)

                qTables = hysteretic(qTables, states, new_actions, alpha, beta, thisReward, gamma, new_states)
                states = new_states

            #print(rewardSum)
            rewardSumInTrial.append(rewardSum)
        
        #print(rewardSumInTrial)
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
    plt.title('Hysteretic')
    plt.xlabel("Trials")
    plt.ylabel("Average Total Reward")
    plt.savefig('./plots/Hysteretic.png')
    plt.clf()

    pkl.dump(qTables[0], open('tables/q-table1.p', 'wb'))
    pkl.dump(qTables[1], open('tables/q-table2.p', 'wb'))


def main():
    trainHysteretic()

if __name__ == '__main__':
    main()