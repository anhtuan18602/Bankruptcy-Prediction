import math
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import levy_stable
from .model import KELM
import matplotlib as plt

def plot_params(cs,ys, best_c,best_y, bounds, labels = None):
    plt.figure(figsize=(8, 6))
        
    plt.scatter(cs,ys, color='blue', marker='o')
    plt.scatter(best_c,best_y, color='red', marker='x',s =100)
    if labels:
        for (i, label) in enumerate(labels):
            plt.annotate(label, (cs[i], ys[i]), textcoords="offset points", xytext=(0,10), ha='center')


    plt.xlim(bounds)
    plt.ylim(bounds)

    plt.xlabel('C')
    plt.ylabel('Gamma')
    plt.title('Scatter Plot of params search')
        

    plt.grid(True)
    plt.show()


class params_opt:
    """
    Class for parameter optimization using cross-validation.
    """
    
    def __init__(self, X, labels, bounds, size_pop):
        self.X = X
        self.labels = labels
        self.bounds = bounds
        self.size_pop = size_pop
    
    def param_opt(self, c, gamma):
        """
        Perform cross-validation to evaluate the performance of a model with given parameters.
        
        Args:
            c: Regularization parameter for the model.
            gamma: Kernel coefficient for the model.
        
        Returns:
            The mean accuracy over the cross-validation folds.
        """
        kf = KFold(n_splits=3, shuffle=True)
        accs = []
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]
            model = KELM(gamma=gamma, C=c)  # Initialize model with given parameters
            ret = model.fit(X_train, y_train)
            if ret != -1:
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                accs.append(acc)
        return np.array(accs).mean()
    
    def is_within_bounds(self, c):
        """
        Check if a parameter is within the specified bounds.
        """
        a, b = self.bounds
        return a <= c <= b
    
    def to_bound(self, c):
        """
        Clamp the parameter to be within the bounds.
        """
        a, b = self.bounds
        if c < a:
            return a
        elif c > b:
            return b
        else:
            return c
            
    def to_bound_O(self, c):
        """
        Adjust parameter by adding a random value within the bounds if it is out of range.
        """
        a, b = self.bounds
        if c < a or c > b:
            return np.random.uniform(a, b)
        else:
            return c

    def choose_S(self, S, S_L, cs, ys, cs_L, ys_L):  
        """
        Update parameters based on comparison of objective scores.
        
        Args:
            S: Current scores.
            S_L: New scores after applying Levy flight.
            cs, ys: Current parameters.
            cs_L, ys_L: New parameters after applying Levy flight.
        
        Returns:
            Updated scores and parameters.
        """
        for i in range(self.size_pop):
            if S[i] < S_L[i]:
                S[i] = S_L[i]
                cs[i] = cs_L[i]
                ys[i] = ys_L[i]
        return S, cs, ys

def LSEOFOA(X, labels, R=1, z=0.2, bounds=(2**(-5), 2**5), max_iters=20, size_pop=10, plot=False):
    """
    Improved Fruit Flies Optimization.
    
    Args:
        X: Feature matrix.
        labels: Target labels.
        R: Range for parameter adjustment.
        z: Probability for random parameter sampling.
        bounds: Tuple defining the lower and upper bounds for parameters.
        max_iters: Maximum number of iterations.
        size_pop: Size of the population.
        plot: Whether to plot the optimization process.

    Returns:
        max_c, max_y: Optimal parameters found.
    """
    cs = np.random.rand(size_pop) * (bounds[1] - bounds[0]) + bounds[0]  # Initialize c parameters
    ys = np.random.rand(size_pop) * (bounds[1] - bounds[0]) + bounds[0]  # Initialize gamma parameters
    obj = params_opt(X, labels, bounds, size_pop)
    vectorized_S = np.vectorize(obj.param_opt)
    vectorized_bound = np.vectorize(obj.to_bound)
    
    # Evaluate initial population
    S = vectorized_S(cs, ys)
    max_c = cs[np.argmax(S)]
    max_y = ys[np.argmax(S)]
    cur_max = np.max(S)

    points_c = []
    points_y = []
    
    for j in range(max_iters):
        # Levy flight
        cs += 2 * R * np.random.rand() - R
        ys += 2 * R * np.random.rand() - R

        cs = vectorized_bound(cs)
        ys = vectorized_bound(ys)
        
        # Generate new parameters using Levy distribution
        cs_L = cs * (1 + levy_stable.rvs(1.5, 0, 1))
        ys_L = ys * (1 + levy_stable.rvs(1.5, 0, 1))
    
        cs_L = vectorized_bound(cs_L)
        ys_L = vectorized_bound(ys_L)
        
        S = vectorized_S(cs, ys)
        S_L = vectorized_S(cs_L, ys_L)
        S, cs, ys = obj.choose_S(S, S_L, cs, ys, cs_L, ys_L)
        
        if np.max(S) > cur_max:
            max_ind = np.argmax(S)
            cur_max = S[max_ind]
            max_c = cs[max_ind]
            max_y = ys[max_ind]
        
        points_c.extend(cs)
        points_y.extend(ys)
        
        # Slime Mould Algorithm (SMA) 
        sorted_inds = np.argsort(S)
        sorted_S = S[sorted_inds]
        sorted_cs = cs[sorted_inds]
        sorted_ys = ys[sorted_inds]
        a = np.arctan(1 - (j / max_iters))
        vb = np.random.uniform(-a, a)
        vc = 1 - j / max_iters
        index_a = np.random.randint(0, size_pop)
        index_b = np.random.randint(0, size_pop)
        csa = cs[index_a]
        ysa = ys[index_a]
        csb = cs[index_b]
        ysb = ys[index_b]
        cs_SMA = []
        ys_SMA = []
        
        for i in range(size_pop):
            p = math.tan(abs(S[i] - cur_max))
            ran = np.random.rand()
            r = np.random.rand()
            sorted_i = np.where(sorted_inds == i)[0]
            
            if ran < z:
                ci = np.random.uniform(bounds[0], bounds[1])
                yi = np.random.uniform(bounds[0], bounds[1])
            else:
                if r < p:
                    if sorted_i < len(sorted_inds) / 2: 
                        Wi = 1 + r * np.log((np.max(S) - S[i]) / (np.max(S) - np.min(S)) + 1)
                    else:
                        Wi = 1 - r * np.log((np.max(S) - S[i]) / (np.max(S) - np.min(S)) + 1)
                    ci = max_c + vb * (Wi * csa - csb)
                    yi = max_y + vb * (Wi * ysa - ysb)
                else:
                    ci = vc * cs[i]
                    yi = vc * ys[i]
                ci = obj.to_bound(ci)
                yi = obj.to_bound(yi)
            cs_SMA.append(ci)
            ys_SMA.append(yi)
        
        cs_SMA = np.array(cs_SMA)
        ys_SMA = np.array(ys_SMA)
        
        S_SMA = vectorized_S(cs_SMA, ys_SMA)
        S, cs, ys = obj.choose_S(S, S_SMA, cs, ys, cs_SMA, ys_SMA)

        if np.max(S) > cur_max:
            max_ind = np.argmax(S)
            cur_max = S[max_ind]
            max_c = cs[max_ind]
            max_y = ys[max_ind]

        points_c.extend(cs)
        points_y.extend(ys)

        # Opposition Elite Based Learning (OEBL)
        cs_O = np.random.rand(size_pop) * (bounds[0] + bounds[1]) - max_c
        ys_O = np.random.rand(size_pop) * (bounds[0] + bounds[1]) - max_y

        vectorized_bound_O = np.vectorize(obj.to_bound_O)
        cs_O = vectorized_bound_O(cs_O)
        ys_O = vectorized_bound_O(ys_O)
        S_O = vectorized_S(cs_O, ys_O)
        S, cs, ys = obj.choose_S(S, S_O, cs, ys, cs_O, ys_O)

        if np.max(S) > cur_max:
            max_ind = np.argmax(S)
            cur_max = S[max_ind]
            max_c = cs[max_ind]
            max_y = ys[max_ind]

        points_c.extend(cs)
        points_y.extend(ys)

    if plot:   
        plot_params(points_c, points_y, max_c, max_y, bounds)

    return max_c, max_y

def calc_X(aj,cs, ys, c, y):
    ce = []
    ye = []
    for e in range(3):
        A = 2 * aj * np.random.rand() - aj
        C = 2 * np.random.rand()
        xc = cs[e] - A * abs(C * cs[e] - c)
        xy = ys[e] - A * abs(C * ys[e] - y)
        ce.append(xc)
        ye.append(xy)
    return np.array(ce).mean(), np.array(ye).mean()


def GWO(X, labels,bounds = (2**(-5), 2**5), max_iters = 20, size_pop = 10, plot = False):
    """
    Grey Wolf Optimization.
    
    Args:
        X: Feature matrix.
        labels: Target labels.
        bounds: Tuple defining the lower and upper bounds for parameters.
        max_iters: Maximum number of iterations.
        size_pop: Size of the population.
        plot: Whether to plot the optimization process.

    Returns:
        max_c, max_y: Optimal parameters found.
    """
    a = np.linspace(2, 0, max_iters)
    cs = np.random.rand(size_pop) * (bounds[1] - bounds[0]) + bounds[0]
    ys = np.random.rand(size_pop) * (bounds[1] - bounds[0]) + bounds[0]
    obj = params_opt(X,labels, bounds, size_pop)
    vectorized_S = np.vectorize(obj.param_opt)
    S = vectorized_S(cs,ys)
    sorted_inds = np.argsort(S)[::-1]
    leaders_S = S[sorted_inds][0:3]
    leaders_cs = cs[sorted_inds][0:3]
    leaders_ys = ys[sorted_inds][0:3]
    points_c = list(cs)
    points_y = list(ys)
    vectorized_bound = np.vectorize(obj.to_bound)
    for j in range(max_iters):
        
        for i in range(size_pop):
            cs[i], ys[i] = calc_X(a[j], leaders_cs, leaders_ys, cs[i], ys[i])
        cs = vectorized_bound(cs)
        ys = vectorized_bound(ys)
        S = vectorized_S(cs,ys)
        sorted_inds = np.argsort(S)[::-1]
        new_leaders_S = S[sorted_inds][0:3]
        new_leaders_cs = cs[sorted_inds][0:3]
        new_leaders_ys = ys[sorted_inds][0:3]
        points_c.extend(cs)
        points_y.extend(ys)
        compare_S = np.concatenate((leaders_S, new_leaders_S))
        sorted_inds = np.argsort(compare_S)[::-1]
        leaders_S = compare_S[sorted_inds][0:3]
        leaders_cs = np.concatenate((leaders_cs, new_leaders_cs))[sorted_inds][0:3]
        leaders_ys = np.concatenate((leaders_ys, new_leaders_ys))[sorted_inds][0:3]
    if plot:   
        plot_params(points_c, points_y, leaders_cs[0],leaders_ys[0], bounds)
    
    return leaders_cs[0],leaders_ys[0]
