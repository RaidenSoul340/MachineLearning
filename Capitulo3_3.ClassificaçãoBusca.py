import pandas as pd
from collections import Counter

# teste inicial: home, busca, logado => comprou
# home, busca
# home, logado
# busca, logado
# busca: 85,71% (7 testes)

df = pd.read_csv('Busca3.csv')

x_df = df[['home', 'busca', 'logado']]
y_df = df['comprou']

Xdummies_df = pd.get_dummies(x_df)
Ydummies_df = y_df

x= Xdummies_df.values
y = Ydummies_df.values

porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1

tamanho_de_treino = int(porcentagem_de_treino * len(y))
tamanho_de_teste = int(porcentagem_de_teste * len(y))
tamanho_de_validacao = len(y) - tamanho_de_treino - tamanho_de_teste

# 0 até 799
treino_dados = x[0:tamanho_de_treino]
treino_marcacoes = y[0  :tamanho_de_treino]

# 800 até 899
fim_de_teste = tamanho_de_treino + tamanho_de_teste
teste_dados = x[tamanho_de_treino:fim_de_teste]
teste_marcacoes = y[tamanho_de_treino:fim_de_teste]

# 900 até 999
validacao_dados = x[fim_de_teste:]
validacao_marcacoes = y[fim_de_teste:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
	modelo.fit(treino_dados, treino_marcacoes)

	resultado = modelo.predict(teste_dados)
	acertos = (resultado == teste_marcacoes)
	total_de_acertos = sum(acertos)
	total_de_elementos = len(teste_dados)

	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

	msg = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto)

	print(msg)

	return taxa_de_acerto

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if resultadoMultinomial > resultadoAdaBoost:
    vencedor = modeloMultinomial
else:
    vencedor = modeloAdaBoost

resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)
total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
print(msg)

# a eficácia do algoritmo que chuta tudo um único valor
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)

#============Anotação============#
#01 => Foi introdusido o comando from sklearn.ensemble import AdaBoostClassifier que tem como
#objetivo melhorar o nosso código e fornoce dados mais seguro;

#02 =>Observe que o AdaBoost acertou 85%, ou seja, 3% a mais que o MultinomialNB. 
#Isso comprova que dependedo do conjunto de dados o nosso teste pode variar, nesse caso, 
# o AdaBoost obteve um resultado melhor que o MultinomialNB para esse conjunto de dados, 
# pode ser que para um outro conjunto de dados, o MultinomialNB tenha um resultado melhor;

#03 => Implantaremos os 2 codigos (MultinomialNB, AdaBoost) para podemos selecionar 
#o melhor entre os 2;





