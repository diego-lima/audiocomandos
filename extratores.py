import numpy as np
from funcoes import *
import statistics

# Toda função que analisar o espectro de frequência vai 
# ignorar as frequências abaixo de corte_passa_altas
corte_passa_altas = 300

TAXA_AMOSTRAGEM = 8000

def dados_na_frequencia(dados):
    """
    Todas funções que analisarem os dados no domínio da frequência
    irão puxar os dados daqui. Isso serve pra garantir que todas estão
    analisando a mesma coisa, mesmo que escolhamos outro fitro de 
    passa-altas ou qualquer coisa
    """
    x, y = get_fft_values(TAXA_AMOSTRAGEM, dados)
    
    # setar y = 0 para todo x < corte_passa_altas
    y = np.array([y[i] if x[i] > corte_passa_altas else 0 for i in range(len(x))])
    
    return x, y
    

def f_media(dados):
    """
    Calcula a média no eixo y dos dados.
    """
    x, y = dados_na_frequencia(dados)
    
    return statistics.mean(y)

def f_centro_gravidade(dados):
    """
    Recebe os dados e retorna o centro de gravidade dos mesmos.
    """
    x, y = dados_na_frequencia(dados)
    
    return calcular_centro_gravidade(x, y)
    
def f_mediana(dados):
    """
    Calcula a mediana no eixo y dos dados.
    """
    x, y = dados_na_frequencia(dados)
    
    return statistics.median(y)

def f_percentil75(dados):
    """
    Calcula o percentil 75 no eixo y dos dados.
    """
    x, y = dados_na_frequencia(dados)
    
    return np.percentile(y, 75)

def f_percentil90(dados):
    """
    Calcula o percentil 90 no eixo y dos dados.
    """
    x, y = dados_na_frequencia(dados)
    
    return np.percentile(y, 90)

def f_percentil99(dados):
    """
    Calcula o percentil 99 no eixo y dos dados.
    """
    x, y = dados_na_frequencia(dados)
    
    return np.percentile(y, 99)

def f_rollof(dados):
    
    x, y = dados_na_frequencia(dados)
    
    return calcular_rollof(y)

def f_fluxo_spectral(dados):
    
    x, y = dados_na_frequencia(dados)
    
    magnitudes_normalizadas = normalize(y)
    
    fluxo = 0
    
    for i, mag in enumerate(magnitudes_normalizadas):
        fluxo = fluxo + (mag - magnitudes_normalizadas[i-1])**2
    
    return fluxo

def f_dominio_tempo(dados):
    
    retorno = 0
    
    for i,dado in enumerate(dados):
        retorno = retorno + (int(np.sign(dado)) - int(np.sign(dados[i-1])))**2

    return retorno


extratores = [
    f_media,
    f_centro_gravidade,
    f_fluxo_spectral,
]