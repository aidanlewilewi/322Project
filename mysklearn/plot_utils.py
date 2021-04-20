import matplotlib.pyplot as plt
import numpy as np
import mysklearn.myutils as utils

def plot_bar_chart(x, y, title, xLabel, yLabel, rotateTicks):
    plt.figure() 
    plt.bar(x,y)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if rotateTicks:
        plt.xticks(rotation=45) 
    plt.show()


def plot_pie_chart(x, y, title):
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.title(title)
    plt.show()    

def plot_histogram(x, title, xLabel, yLabel):
    plt.figure()
    plt.hist(x)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def plot_scatter(x,y, title, xLabel, yLabel):
    plt.figure()
    plt.scatter(x,y)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    m, b = compute_slope_intercept(x,y)
    cov = round(utils.compute_covariance(x,y), 2)
    r = round(utils.compute_correlation_coefficient(x,y), 2)
    plt.annotate('r = ' + str(r), xy = (0.6, 0.8), xycoords = 'axes fraction')
    plt.annotate('cov = ' + str(cov), xy = (0.6, 0.9), xycoords = 'axes fraction')

    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c='r', lw=3)



def compute_slope_intercept(x, y):
    xBar = np.mean(x)
    yBar = np.mean(y)

    slope = sum((x - xBar) * (y - yBar)) / sum((x - xBar) ** 2)

    b = yBar - xBar * slope

    return slope, b


def plot_scatter_xy(x,y, title, xLabel, yLabel, xpos, ypos):
    plt.figure()
    plt.scatter(x,y)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    m, b = compute_slope_intercept(x,y)
    cov = round(utils.compute_covariance(x,y), 2)
    r = round(utils.compute_correlation_coefficient(x,y), 2)
    plt.annotate('r = ' + str(r), xy = (xpos, ypos), xycoords = 'axes fraction')
    plt.annotate('cov = ' + str(cov), xy = (xpos, ypos + 0.1), xycoords = 'axes fraction')

    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c='r', lw=3)