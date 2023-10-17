import pandas as pd

df = pd.read_csv("Busca.csv")
x_df = (df[["home", "busca", "logado"]])
y_df = (df["comprou"])
#Se usar o codigo print(df["home"]) ira mostra apenas dados da coluna.

#As colunas são impressas, porém, ainda existe um pequeno detalhe... Analisando a coluna busca, 
# vemos que os valores ainda são as variáveis categóricas! Lembra que precisamos utilizar as 
# categorias da coluna busca? Ou seja, precisamos pegar os dummies dessa coluna. Como faremos isso? 
# Podemos pedir para pandas devolver os dummies do nosso X
Xdummies_df = pd.get_dummies(x_df).astype(int)
Ydummies_df = y_df

x = Xdummies_df.values
y = Ydummies_df.values

# a eficácia do algoritmo que chuta tudo 0 ou 1
acerto_de_um = len(y[y==1])
acerto_de_zero = len(y[y==0]) - acerto_de_um
#o nosso algoritmo base precisa utilizar o maior resultado para o mesmo chute. 
# Porém, como podemos pegar a maior variável entre elas? Simples, no python, 
# podemos utilizar a função max() enviando duas variáveis como parâmetro, então, 
# ele retorna o valor maior
taxa_de_acerto_base = 100.0 * max(acerto_de_um, acerto_de_zero) / len(y)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

porcentagem_treino = 0.9

tamanho_de_treino = 0.9 * len(y)
#90% dos dados(primeiros)
tamanho_de_teste = 0.1 * len(y)
#10% dos dados(ultimos)

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

diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(taxa_de_acerto)
print(total_de_elementos)



