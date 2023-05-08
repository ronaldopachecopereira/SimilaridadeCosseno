import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import math

# carregar dados do arquivo CSV
dados = pd.read_excel('Luxurywatch.xlsx')

# selecionar as colunas relevantes para comparação
dados_selecionados = dados[['Brand', 'Model', 'Case_mat_strap', 'Type', 'price']]

# transformar os dados em um formato de texto unificado para vetorização
dados_texto = dados_selecionados.apply(lambda x: ' '.join(x.astype(str)), axis=1)

# criar o vetorizador TF-IDF
vetorizador = TfidfVectorizer()

# vetorizar as descrições dos produtos
vetores = vetorizador.fit_transform(dados_texto)

# exemplo de relógio fornecido pelo usuário
relogio_usuario = input('Digite a marca, modelo ou material do relógio que você quer verificar a similaridade (separados por espaço): ').split()

# transformar o exemplo em um vetor numérico
relogio_usuario_vetor = vetorizador.transform([' '.join(map(str, relogio_usuario + [''] * (len(dados_selecionados.columns) - len(relogio_usuario))))])

# calcular a similaridade do cosseno entre o relógio fornecido pelo usuário e todos os outros relógios
similaridades = cosine_similarity(relogio_usuario_vetor, vetores)

# obter os índices dos relógios ordenados por ordem decrescente de similaridade
indices_similares = similaridades.argsort()[0][::-1]

# selecionar os 10 relógios mais similares, excluindo aqueles com a mesma marca que o exemplo dado
marca_usuario = relogio_usuario[0]
top_similares = []
for i in indices_similares:
    if dados.loc[i, 'Brand'] != marca_usuario:
        top_similares.append(i)
    if len(top_similares) >= 10:
        break

# imprimir os 10 relógios mais similares
print("Os 10 relógios mais similares a", ' '.join(map(str, relogio_usuario)), "são:")

for i in top_similares:
    print(dados.loc[i, ['Brand', 'Model', 'Case_mat_strap', 'Type', 'price']])
# adicionar as colunas de similaridade do cosseno e ângulo do cosseno
dados_selecionados.loc[:, 'Cosine Similarity'] = similaridades[0]
dados_selecionados.loc[:, 'Cosine Angle'] = [math.degrees(math.acos(similarity)) for similarity in similaridades[0]]

# imprimir os 10 Relogios mais similares com as colunas adicionais
print(dados_selecionados.loc[top_similares, ['Brand', 'Model', 'price', 'Cosine Similarity', 'Cosine Angle']])
# extrair os vetores dos 10 relógios mais similares
vetores_similares = vetores[top_similares]

# calcular os ângulos dos vetores
angulos = []
for vetor in vetores_similares:
    angulo = np.arccos(cosine_similarity(relogio_usuario_vetor, vetor)[0][0]) * 180 / np.pi
    angulos.append(angulo)
# extrair as coordenadas dos vetores dos 10 relógios mais similares
coordenadas_similares = vetores_similares.toarray()
# adicionar as colunas de similaridade do cosseno e ângulo do cosseno
dados_selecionados['Cosine Similarity'] = similaridades[0]
dados_selecionados['Cosine Angle'] = [math.degrees(math.acos(similarity)) for similarity in similaridades[0]]
#print("Os relogios mais similares a ", ' '.join(map(str, relogio_usuario)), "são:")

# extrair as coordenadas dos vetores dos 10 relógios mais similares
coordenadas_similares = vetores_similares.toarray()

plt.show()