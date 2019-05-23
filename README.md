# audiocomandos
Projeto: reconhecer comandos em trechos de audios (como assobio ou palmas)

Vamos processar faixas de áudio (extraindo features a partir da análise do espectro de frequência) para identificar comandos sonoros.

Vamos treinar um Multilayer Perceptron (MLP) para fazer esse reconhecimento.

O processo é o seguinte: começamos fazendo nossa [própria implementação de uma MLP](https://nbviewer.jupyter.org/github/diego-lima/audiocomandos/blob/master/MLP.ipynb).

Coletamos alguns áudios, e fizemos [uma análise](https://nbviewer.jupyter.org/github/diego-lima/audiocomandos/blob/master/Analisando%20audio.ipynb) nesses áudios, para saber como poderíamos processá-los, antes de jogar na MLP.

Por fim, geramos um dataset (fizemos nossa própria coleta de exemplos) e o utilizamos para [treinar a MLP](https://nbviewer.jupyter.org/github/diego-lima/audiocomandos/blob/master/Modelo.ipynb).


[Essa](https://github.com/diego-lima/audiocomandos/blob/master/IA%20Audiocomandos.pdf) foi a apresentação desse projeto na disciplina de IA.


Uma descrição rápida do conteúdo desse repositório:

1. audios/
    - Contém algumas faixas de audio usadas para a brincadeira inicial de analisar o áudio
2. csv/
    - Contém os dados gerados a partir da análise dos áudios de exemplo coletados (estes são os dados que jogamos na MLP)
3. dataset/
    - Contém os áudios de exemplo coletados, separados por classe em cada pasta
4. MLP.py
    - Contém o código python extraído de MLP.ipynb. Essa extração é feita programaticamente, e serve para podermos utilizar as funções que tem lá em outros cantos
5. coletor.py
    - Um utilitário que ajuda a gravar os arquivos de áudio de exemplo já separados em pastas
6. funcoes.py
    - Funções úteis, como cálculo de centro de gravidade, pegar o nome do arquivo mais recente em um diretório, etc
