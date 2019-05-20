import os
import pickle
import queue
import pyaudio
import pandas as pd
import threading
import logging
import time


from funcoes import arquivo_mais_recente
from MLP import predizer
from extratores import *

"""
SETUP
"""

raiz_pesos = 'rede'
raiz_audios = 'dataset'

# Carregando pesos
with open(arquivo_mais_recente(raiz_pesos), 'rb') as myf:
    rede = pickle.load(myf)

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
        fila.put(np.fromstring(data, dtype=np.int16))
    
coleta = threading.Thread(target=coletar_audio)
coleta.start()

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
# coleta.start()

atualizacao_buffer = threading.Thread(target=atualizar_buffer)
# atualizacao_buffer.start()


"""
TRATANDO OS DADOS PRA JOGAR NO MLP
"""

# dado = {f.__name__: f(dados) for f in extratores}
# dado = pd.DataFrame([dado])
print(extratores)

stream.stop_stream()
stream.close()
p.terminate()