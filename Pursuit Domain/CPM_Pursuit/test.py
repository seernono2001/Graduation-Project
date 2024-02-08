import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pandas as pd


def pic(data):

    Y = list(range(len(data)))
    plt.scatter(Y,data,color="red")
    plt.title("Scatter Plot of the data")
    plt.xlabel("X")
    plt.ylabel("Y")

    linear_model=np.polyfit(Y,data,2)
    linear_model_fn=np.poly1d(linear_model)
    x_s=np.arange(0,5000)
    plt.plot(x_s,linear_model_fn(x_s),color="green")

    plt.show()
    
def main():
    with open('D:/專題/code/Pursuit Domain/Hysteretic_Pursuit/tables/captures.p', 'rb') as file:
        captures_Hysteretic = pkl.load(file)
    with open('D:/專題/code/Pursuit Domain/Hysteretic_Pursuit/tables/steps.p', 'rb') as file:
        steps_Hysteretic = pkl.load(file)
    
    pic(captures_Hysteretic)


if __name__ == '__main__':
    main()