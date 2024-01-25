import numpy as np

def __main__():
    x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
    y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

    meanWPlus, meanLPlus = np.mean(x_train[y_train == 1], axis = 0)
    meanWMinus, meanLMinus = np.mean(x_train[y_train == -1], axis = 0)

    varWPlus, varLPlus = np.var(x_train[y_train == 1], axis = 0)
    varWMinus, varLMinus = np.var(x_train[y_train == -1], axis = 0)

    print("Means: ", meanWPlus, meanLPlus, meanWMinus, meanLMinus)
    print("Var: ", varWPlus, varLPlus, varWMinus, varLMinus)

    x = [40, 10] # to predict
    aMinus = lambda x: -(x[0] - meanLMinus) ** 2 / (2 * varLMinus) - (x[1] - meanWMinus) ** 2 / (2 * varWMinus)
    aPlus = lambda x: -(x[0] - meanLPlus) ** 2 / (2 * varLPlus) - (x[1] - meanWPlus) ** 2 / (2 * varWPlus)
    y = np.argmax([aMinus, aPlus])

    print("Class:", y)
   

if __name__ == '__main__':
    __main__()

