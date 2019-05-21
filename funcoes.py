from typing import Union
import numpy as np
import os
import glob
import time
import hashlib

from scipy.fftpack import fft

def arquivo_mais_recente(diretorio="."):
    """
    Recebe o caminho de uma pasta e retorna o nome do arquivo mais recente.
    """
    files = os.listdir(diretorio)
    paths = [os.path.join(diretorio, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def nome_diferente(raiz, caminho='.', sufixo=1, formato='wav', tem_hash=True):
    """ Recebe um nome que você quer que o arquivo tenha.
    Retorna um nome seguido de um número de forma que o nome retornado não exista ainda no diretório.

    Se sua raiz for 'palma', mas já existir 'palma_1.wav', eu vou retornar 'palma_2.wav'

    raiz:       nome do arquivo
    diretorio:  onde procurar se já tem um arquivo existente com nome igual
    sufixo:     primeiro inteiro que devemos verificar se já tem, e ir incrementando enquanto houver
    formato:    formato do arquivo gerado
    tem_hash:   se deve incluir um hashzinho no nome do arquivo, para prevenir colisão em merge
    """

    nomes_existentes = os.listdir(caminho)

    i = sufixo
    while True:
        if tem_hash:
            novo_nome = "%s-%d" % (raiz, i)
            seed = "%s%d" % (novo_nome, time.time())
            seed = seed.encode('utf-8')
            seed = hashlib.md5(seed).hexdigest()[:6]
            novo_nome = "%s-%s.%s" % (novo_nome, seed, formato)
        else:
            novo_nome = "%s-%d.%s" % (raiz, i, formato)

        if novo_nome in nomes_existentes:
            i += 1
            continue

        if caminho == '.':
            return novo_nome

        return os.path.join(caminho, novo_nome)

    
def get_fft_values(sampling_rate, data):
    size = data.shape[0]
    f_values = np.linspace(0.0, sampling_rate/2.0, size//2)
    fft_values_ = fft(data)
    fft_values = 2.0/size * np.abs(fft_values_[0:size//2])
    return f_values, fft_values


def calcular_centro_gravidade(indices: np.array, valores: np.array):
    """Pega uma série numérica e retorna seu centro de gravidade"""

    return (indices*valores).sum() / valores.sum()


def calcular_rollof(valores: np.array):

    return 0.8*valores.sum()


def normalize(d):
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)

    return d

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
