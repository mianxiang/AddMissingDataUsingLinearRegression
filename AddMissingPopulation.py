import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

def get_data(filename):
    data = pd.read_csv(filename)
    X_parameter = []
    Y_parameter = []
    for year, num in zip(data["Year"], data["Num"]):
        X_parameter.append([float(year)])
        Y_parameter.append(float(num))

    return X_parameter, Y_parameter

def linear_model_main(X_parameter, Y_parameter, predict_value):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameter, Y_parameter)
    predict_outcome = regr.predict(predict_value)
    return predict_outcome

if __name__ == "__main__":
    x,y = get_data("Population.csv")
    population_1999 = linear_model_main(x, y, 1999)
    population_2000 = linear_model_main(x, y, 2000)
    population_2001 = linear_model_main(x, y, 2001)

    print("The missing population in {} is {}".format(1999, population_1999))
    print("The missing population in {} is {}".format(2000, population_2000))
    print("The missing population in {} is {}".format(2001, population_2001))