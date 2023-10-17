import pandas as pd
from collections import Counter

df = pd.read_csv('Busca2.csv')
x_df = df[['home', 'busca', 'logado']]
y_df = df['comprou']

Xdummies_df = pd.get_dummies(x_df).astype(int)
Ydummies_df = y_df

x = Xdummies_df.values
y = Ydummies_df.values

porcentagem_treino = 0.9

tamanho_de_treino = int(porcentagem_treino * len(y))
tamanho_de_teste = len(y) - tamanho_de_treino

treino_dados = x[:tamanho_de_treino]
treino_marcacoes = y[:tamanho_de_treino]

teste_dados = x[-tamanho_de_teste:]
teste_marcacoes = y[-tamanho_de_teste:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

acertos = resultado == teste_marcacoes

total_de_acertos = sum(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print("Taxa de acerto do algoritmo: %f" % taxa_de_acerto)
print(total_de_elementos) 

acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(teste_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)