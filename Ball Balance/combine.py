import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def picture(x_h, v_h, x_w, v_w):
    plt.plot(x_h, '-', label="Hysteretic Space", color='blue')
    plt.plot(x_w, '-', label="Wolf Space", color='red')
    plt.axhline(0, color='black', linestyle='--', label="y=0")
    plt.legend()
    plt.title("Space")
    plt.savefig("D:/專題/code/Ball Balance/" + "Compare" + "_Space.png")
    plt.clf()

    plt.plot(v_h, '-', label="Hysteretic Speed", color='blue')
    plt.plot(v_w, '-', label="Wolf Speed", color='red')
    plt.axhline(0, color='black', linestyle='--', label="y=0")
    plt.legend()
    plt.title("Speed")
    plt.savefig("D:/專題/code/Ball Balance/" + "Compare" + "_Speed.png")
    plt.clf()



def main():
    with open('D:/專題/code/Ball Balance/hysteretic_Qlearning/tables/space.p', 'rb') as file:
        space_Hysteretic = pkl.load(file)
    with open('D:/專題/code/Ball Balance/hysteretic_Qlearning/tables/velocities.p', 'rb') as file:
        v_Hysteretic = pkl.load(file)
    with open('D:/專題/code/Ball Balance/Wolf/tables/space.p', 'rb') as file:
        space_Wolf = pkl.load(file)
    with open('D:/專題/code/Ball Balance/Wolf/tables/velocities.p', 'rb') as file:
        v_Wolf = pkl.load(file)

    picture(space_Hysteretic, v_Hysteretic, space_Wolf, v_Wolf)

if __name__ == '__main__':
    main()