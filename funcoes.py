from typing import Union
import numpy as np


def calcular_centro_gravidade(indices: np.array, valores: np.array):
    """Pega uma série numérica e retorna seu centro de gravidade"""

    return (indices*valores).sum() / valores.sum()


def calcular_rollof(valores: np.array):

    return 0.8*valores.sum()


def normalize(d):
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)

    return d


def func_signal(d):

    if d > 0:
        return -1
    elif d == 0:
        return 0
    else:
        return -1

def sigmoid(z):
    '''
    Sigmoid function
    z can be an numpy array or scalar
    '''
    result = 1.0 / (1.0 + np.exp(-z))
    return result

def sigmoid_derivative(z):
    '''
    Derivative for Sigmoid function
    z can be an numpy array or scalar
    '''
    result = sigmoid(z) * (1 - sigmoid(z))
    return result

def relu(z):
    '''
    Rectified Linear function
    z can be an numpy array or scalar
    '''
    if np.isscalar(z):
        result = np.max((z, 0))
    else:
        zero_aux = np.zeros(z.shape)
        meta_z = np.stack((z , zero_aux), axis = -1)
        result = np.max(meta_z, axis = -1)
    return result
    

def relu_derivative(z):
    '''
    Derivative for Rectified Linear function
    z can be an numpy array or scalar
    '''
    result = 1 * (z > 0)
    return result

    
def memoize(func):
    '''
    Serve para fazer um cache dos resultados de uma função.
    '''
    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func
