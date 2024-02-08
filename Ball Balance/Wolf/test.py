from functions import *
import matplotlib.pyplot as plt
import pickle as pkl

d_win = 0.0025
d_lose = 0.01


def test(qTables, P1, P2):
    spaces, velocities = [], []
    states = (0.495, 1.041)
    spaces.append(states[0])
    velocities.append(states[1])

    # print(P1)
    # print(P2)
    for i in range(30):
        a1 = actions_select(states, P1, None, qTables[0])  # 调用agent实例的act方法
        a2 = actions_select(states, P2, None, qTables[1])

        actions = [a1, a2]

        x, v = nextState(h1=actions[0], h2=actions[1], v=states[1], t=0.03, x_0=states[0], v_0=states[1])
        spaces.append(x)
        velocities.append(v)
        # if the ball falls from the flat surface
        if np.abs(x) > 1:
            break

        new_states = (np.round(x, decimals=3), np.round(v, decimals=3))
        new_states = isValidState(new_states)

        states = new_states

    pkl.dump(spaces, open('tables/space.p', 'wb'))
    pkl.dump(velocities, open('tables/velocities.p', 'wb'))
    plt.plot(spaces, "-", label="Space", color="black")
    plt.plot(velocities, "-", label="Speed", color="red")
    plt.axhline(0, color='blue', linestyle='--', label="y=0")
    plt.legend()
    plt.title("Wolf")
    plt.savefig("./plots/" + "Wolf" + "_test.png")
    plt.clf()


def main():
    with open("./tables/q-table1.p", "rb") as file:
        qTable1 = pkl.load(file)
    with open("./tables/q-table2.p", "rb") as file:
        qTable2 = pkl.load(file)
    with open("./tables/policy-table1.p", "rb") as file:
        P1 = pkl.load(file)
    with open("./tables/policy-table2.p", "rb") as file:
        P2 = pkl.load(file)
    qTables = [qTable1, qTable2]

    test(qTables,P1, P2)


if __name__ == "__main__":
    main()
