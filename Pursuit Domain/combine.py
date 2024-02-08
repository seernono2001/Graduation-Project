import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.signal import savgol_filter


def picture(h, w, type):

    trial = list(range(len(h)))

    # 绘制数组值的范围，使用淡色表示
    plt.plot(h, '-', label=None, color='lightcoral')
    plt.plot(w, '-', label=None, color='lightblue')

    yhat_h = savgol_filter(h, 51, 3)
    yhat_w = savgol_filter(w, 51, 3)

    plt.plot(trial,yhat_h, color='red', label = "Hysteretic")
    plt.plot(trial,yhat_w, color='blue', label = "Wolf")

    plt.legend()
    plt.xlabel('Trials')

    if(type == "c"):
        # 设置图形标题和轴标签
        plt.ylabel('Captures')

        plt.title("Average Captures")
        plt.savefig("D:/專題/code/Pursuit Domain/" + "Compare" + "_Captures.png")
    else:
        # 设置图形标题和轴标签
        plt.ylabel('Steps')

        plt.title("Steps")
        plt.savefig("D:/專題/code/Pursuit Domain/" + "Compare" + "_Steps.png")
        
    plt.clf()
 


def main():
    with open('D:/專題/code/Pursuit Domain/Hysteretic_Pursuit/tables/captures.p', 'rb') as file:
        captures_Hysteretic = pkl.load(file)
    with open('D:/專題/code/Pursuit Domain/Hysteretic_Pursuit/tables/steps.p', 'rb') as file:
        steps_Hysteretic = pkl.load(file)
    with open('D:/Wolf_Pursuit/captures.p', 'rb') as file:
        captures_Wolf = pkl.load(file)
    with open('D:/Wolf_Pursuit/steps.p', 'rb') as file:
        steps_Wolf = pkl.load(file)

    picture(captures_Hysteretic, captures_Wolf, type = "c")
    picture(steps_Hysteretic, steps_Wolf, type = "s")

if __name__ == '__main__':
    main()