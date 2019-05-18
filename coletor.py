#!/usr/bin/env python3
"""Grava arquivos de áudio num diretório.
"""

import argparse
import os
import time
import subprocess

"""
CONSTANTES
"""
# as constantes abaixo são pra printar cores no terminal
ON_ESCAPE = "\033[1;37;42m"
OFF_ESCAPE = "\033[1;30;41m"
RESET_ESCAPE = '\x1b[0m'

# dimensões do terminal do usuário
COL_TERMINAL, LIN_TERMINAL = os.get_terminal_size(0)


"""
FUNÇÕES
"""


def on(texto="  ON  "):
    """Retorna um texto pra ser printado com fundo verde"""
    return ON_ESCAPE + texto + RESET_ESCAPE


def off(texto="  OFF "):
    """Retorna um texto pra ser printado com fundo vermelho"""
    return OFF_ESCAPE + texto + RESET_ESCAPE


def nome_diferente(raiz, caminho='.', sufixo=1, formato='wav'):
    """ Recebe um nome que você quer que o arquivo tenha.
    Retorna um nome seguido de um número de forma que o nome retornado não exista ainda no diretório.

    Se sua raiz for 'palma', mas já existir 'palma_1.wav', eu vou retornar 'palma_2.wav'

    raiz:       nome do arquivo
    diretorio:  onde procurar se já tem um arquivo existente com nome igual
    sufixo:     primeiro inteiro que devemos verificar se já tem, e ir incrementando enquanto houver
    formato:    formato do arquivo gerado
    """

    nomes_existentes = os.listdir(caminho)

    i = sufixo
    while True:
        novo_nome = "%s-%d.%s" % (raiz, i, formato)

        if novo_nome in nomes_existentes:
            i += 1
            continue

        if caminho == '.':
            return novo_nome

        return os.path.join(caminho, novo_nome)


"""
PARSER
"""
parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    'diretorio', type=str, default='audio', nargs='?',
    help='diretório onde guardar os arquivos de audio. O padrão é "%(default)s".')

parser.add_argument(
    '-t', '--tempo', type=float, default=2,
    help='Tempo em segundos de cada arquivo de audio gravado. O padrão é %(default)s.')

parser.add_argument(
    '-p', '--parar', type=float, default=2,
    help='Tempo em segundos que paramos de gravar entre cada arquivo de audio gravado. O padrão é %(default)s.')

parser.add_argument(
    '-q', '--quantidade', type=int, default=5,
    help='Quantos arquivos de audio vamos querer gravar. O padrão é %(default)s.')

parser.add_argument(
    '-n', '--nome', type=str, default=None,
    help='Nome dos arquivos que serão gravados.' +
         'O padrão o último nome do diretório (ex: "audio/palmas" vai dar "palmas")')

"""
SCRIPT
"""

args = parser.parse_args()

diretorio = args.diretorio

nome = args.nome
nome = nome if nome else diretorio.split('/')[-1]

quantidade = 1

if not os.path.exists(diretorio):
    os.makedirs(diretorio)

while quantidade <= args.quantidade:
    # Printar ON quantidade/total
    s = "%s %d/%d" % (on(), quantidade, args.quantidade)
    print(s, end='\r')

    novo_arquivo = nome_diferente(nome, diretorio)
    print("\t\t\t%s" % novo_arquivo, end='\r')

    p = subprocess.Popen(["arecord", novo_arquivo], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(args.tempo)

    p.terminate()

    if quantidade < args.quantidade:
        # Printar OFF quantidade/total
        s = "%s %d/%d" % (off(), quantidade, args.quantidade)
        print(s, end='\r')
        time.sleep(args.parar)
    else:
        # Printar OFF ACABOU
        s = "%s ACABOU" % off()
        print(s, end='\r')
        time.sleep(2)

    quantidade += 1

print(" " * COL_TERMINAL, end='\r') # para limpar a linha inteira
print("Arquivos de áudio salvos em '%s'" % diretorio)
