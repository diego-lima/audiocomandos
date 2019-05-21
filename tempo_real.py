import os
import pickle
import queue
import pyaudio
import pandas as pd
import threading
import logging
import time
import matplotlib.pyplot as plt


from funcoes import arquivo_mais_recente, colorido
from MLP import predizer
from extratores import *

CORES = [
    "\033[1;37;42m", #verde
    "\033[1;37;45m", #purple
    "\033[1;30;41m", #vermelho
    "\033[1;37;44m", #azul
    "\033[1;37;43m" #amarelo
]


extratores = [
    f_media,
    f_centro_gravidade,
#     f_mediana,
#     f_percentil75,
#     f_percentil90,
    f_percentil99,
    f_fluxo_spectral,
#    f_dominio_tempo,
]

"""
SETUP
"""

raiz_pesos = 'rede'
raiz_audios = 'dataset'

# Carregando informações
with open(arquivo_mais_recente(raiz_pesos), 'rb') as myf:
    bundle = pickle.load(myf)

rede = bundle['rede']
treino_media = bundle['treino_media']
treino_desvio_padrao = bundle['treino_desvio_padrao']
classes = bundle['classes']

# deixar os extratores na ordem certa
copia = extratores[:]
for i,feature in enumerate(treino_desvio_padrao.index):
    busca = [f for f in copia if f.__name__ == feature]
    extratores[i] = busca[0]

# Nomes das classes
classes = os.listdir(raiz_audios)

# Configurações da leitura de áudio
janela_tempo_segundos = 2
taxa_amostragem = 8000
bloco_amostras = 1000

# variáveis globais
fila = queue.Queue()
audiodata = np.zeros(janela_tempo_segundos * taxa_amostragem)
threads_ligadas = True

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=taxa_amostragem, input=True, frames_per_buffer=bloco_amostras)

# threads

def coletar_audio():
    global fila
    global stream
    global controle_threads
    
    while threads_ligadas:
        data = stream.read(bloco_amostras)
        fila.put(np.frombuffer(data, dtype=np.int16))
    
def atualizar_buffer():
    global audiodata
    global fila
    while threads_ligadas:
        while True:
            try:
                data = fila.get_nowait()
            except queue.Empty:
                break

            audiodata = np.roll(audiodata, -bloco_amostras)
            audiodata[-bloco_amostras:] = data
        

coleta = threading.Thread(target=coletar_audio)

atualizacao_buffer = threading.Thread(target=atualizar_buffer)

coleta.start()
atualizacao_buffer.start()


"""
TRATANDO OS DADOS PRA JOGAR NO MLP
"""

while True:
    # Copiar os últimos 2 segundos de áudio
    dados = audiodata.copy()

    # Aplicar os extratores de features
    dado = [f(dados) for f in extratores]

    # Normalizar para aquelas features
    for i in range(len(dado)):
        dado[i] = (dado[i] - treino_media[i]) / treino_desvio_padrao[i]

    resultado = predizer(rede, dado)
    print(colorido(CORES[resultado],"     "), "    (%s)" % classes[resultado])


"""
DESLIGANDO
"""

threads_ligadas = False
time.sleep(1)

stream.stop_stream()
stream.close()
p.terminate()