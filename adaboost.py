# -*- coding: latin-1 -*-

# https://www.lri.fr/~antoine/Courses/Master-ISI/Tr-boosting-06x4.pdf

# https://www.lri.fr/~antoine/Courses/Master-ISI/section-boosting.pdf
from __future__ import division
import numpy as np

from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import genfromtxt
import csv
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

"""
AdaBoost
https://gist.github.com/tristanwietsma/5486024
"""
"""
class AdaBoost:

    def __init__(self, training_set):
        self.training_set = training_set
        self.N = len(self.training_set)
        self.weights = np.ones(self.N)/self.N # array([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
        self.RULES = []
        self.ALPHA = []

    def set_rule(self, func, test=False):
        errors = np.array([t[1]!=func(t[0]) for t in self.training_set]) # array([False, False,  True,  True,  True, False, False, False, False, False], dtype=bool) it checks if input is not equal to output i.e. error
        # print self.training_set,errors
        e = (errors*self.weights).sum()
        if test:
            return e
        alpha = 0.5 * np.log((1-e)/e) # Here e is negatively proportional to alpha
        print 'e=%.2f a=%.2f'%(e, alpha)
        w = np.zeros(self.N) # array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        for i in range(self.N):
            # Increase/Decrease weight based on if error(misclassified) ocurred
            if errors[i] == 1:
                w[i] = self.weights[i] * np.exp(alpha) # Increase if error found
            else:
                w[i] = self.weights[i] * np.exp(-alpha) # Decreased if correctly classifies
        self.weights = w / w.sum() # Changed weights; It will affect the value of e and thusly alpha
        self.RULES.append(func)
        self.ALPHA.append(alpha)

    def evaluate(self):
        NR = len(self.RULES)
        for (x,l) in self.training_set:
            hx = [self.ALPHA[i]*self.RULES[i](x) for i in range(NR)]
            print x, np.sign(l) == np.sign(sum(hx))
"""

"""
This version is based on Patrick Winston's lecture of Boosting (https://www.youtube.com/watch?v=UHBmv7qCey4).
It turns out you do not have to calculate the error and do the exponential of the alpha.
Instead, it is mathematically equivalent to just making sure that the weights for the correct predictions add up to 1/2 and the weights of the incorrect predictions add up to 1/2.
I've included the alpha and error calculations above just to show that the weights are the same as the original code, but feel free to remove them and save some time.
"""
class AdaBoostHalf:

    def __init__(self, training_set):
        self.training_set = training_set
        self.N = len(self.training_set)
        self.weights = np.ones(self.N)/self.N
        self.RULES = []
        self.ALPHA = []

    def set_rule(self, func, test=False):
        errors = np.array([t[1]!=func(t[0]) for t in self.training_set]) # Evaluation de la regle de decision
        print("Func", func)
        print("Errors", errors)

        e = (errors*self.weights).sum()
        if test:
            return e
        alpha = 0.5 * np.log((1-e)/e)  # Note, this line can be deleted since we are using the 1/2 trick
        print ('e={} a={}'.format(e, alpha))
        w = np.zeros(self.N)
        sRight = np.sum(self.weights[errors])
        sWrong = np.sum(self.weights[~errors])
        for i in range(self.N):
            if errors[i] == 1:
                w[i] = self.weights[i]/(2.0*sRight)
            else:
                 w[i] = self.weights[i]/(2.0*sWrong)

        self.weights = w / w.sum()
        print(self.weights)
        self.RULES.append(func)
        self.ALPHA.append(alpha)
        print('alpha = {}'.format(alpha))

    def evaluate(self):
        NR = len(self.RULES)
        for (x,l) in self.training_set:
            hx = [alpha * rules(x) for alpha, rules in zip(self.ALPHA, self.RULES)]
            print (x, np.sign(l) == np.sign(sum(hx)))


def functionname(x):
    # Ici on met le modele predictif
    # retourne 1 si vrai / -1 si faux
    print("functionname",x )
    return 1

if __name__ == '__main__':

    examples = []
    """
    examples.append(([[22.82389868, 24.12099087, 25.44874231]], 1))
    examples.append(([[52.8238, 24.12099087, 2245.44874231]], -1))
    examples.append(([[656.89868, 224.12099087, 25.44874231]], -1))
    """


    #examples.append(((1,  2  ), 1))
    examples.append(((1,  4  ), 1))
    examples.append(((2.5,5.5), 1))
    examples.append(((3.5,6.5), 1))
    examples.append(((4,  5.4), 1))
    examples.append(((2,  1  ),-1))
    examples.append(((2,  4  ),-1))
    examples.append(((3.5,3.5),-1))
    examples.append(((5,  2  ),-1))
    examples.append(((5,  5.5),-1))


    m = AdaBoostHalf(examples)
    """
    m.set_rule(lambda x: functionname(x))
    m.set_rule(lambda x: functionname(x))
    m.set_rule(lambda x: functionname(x))
    m.set_rule(lambda x: functionname(x))
    """
    #m.set_rule(lambda x: 2*(x[0] < 1.5)-1) # Here some thresolds are used to form a sort of decision tree
    m.set_rule(lambda x: 2*(x[0] < 4.5)-1)
    m.set_rule(lambda x: 2*(x[1] > 5)-1)

    m.evaluate()
