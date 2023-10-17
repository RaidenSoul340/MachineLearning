from dados import carregar_acessos

x, y = carregar_acessos()

#Precisamos adicionar 90% para ambos, então as 90 primeiras linhas para cada um deles:
treino_dados = x[:90]
treino_marcacoes = y[:90]
#Agora precisamos adicionar os 10% que restaram para as variáveis de teste, porém, dessa vez, 
# precisamos das 9 últimas linhas!:
teste_dados = x[-9:]
teste_marcacoes = y[-9:]


from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

diferenca = resultado - teste_marcacoes

acertos = [d for d in diferenca if d == 0]

total_de_acertos = len(acertos)
#O total de acerto era justamente o tamanho da nossa variável acertos, 
# pois ela representa a quantidade de acertos que tivemos
total_de_elementos = len(teste_dados)
#total de elementos é a quantidade de dados que nós temos, ou seja, o tamanho do nosso 

taxa_de_acertos = [100.0 * total_de_acertos / total_de_elementos]

print(taxa_de_acertos)
print(total_de_elementos)



