from sklearn.naive_bayes import MultinomialNB

# [É gordinho?, Tem perna curta?, faz au-au?]
porco1 = [1,1,0]
porco2 = [1,1,0]
porco3 = [1,1,0]
cachorro4 = [1,1,1]
cachorro5 = [0,1,1]
cachorro6 = [0,1,1]

dados = [porco1, porco2, porco3, cachorro4, cachorro5, cachorro6]

marcadores = [1,1,1,0,0,0]

marcadores_testes = [0,1,0]

misterioso1 = [1,1,1]
misterioso2 = [1,0,0]
misterioso3 = [0,0,1]

teste  = (misterioso1, misterioso2, misterioso3)

modelo = MultinomialNB()
modelo.fit(dados, marcadores)

marcacoes_teste = [1, 1, 0]

resultado = modelo.predict(teste)

diferencas = resultado - marcadores_testes

acertos = [d for d in diferencas if d == 0]

total_de_acertos = len(acertos)

total_de_elementos = len(teste)

taxa_de_acertos = [100.0 * total_de_acertos / total_de_elementos]

print(resultado)
print(diferencas)
print(taxa_de_acertos)

#Resultado e marcação: 1, resultará em: 1 - 1 = 0.
#Resultado: 1 e marcação: -1, resultará em: 1 - (-1) -> 1 + 1 = 2.
#Resultado: -1 e marcação: -1, resultará em: -1 + 1 = 0.
#Resultado: -1 e marcação: 1, resultará em: -1 - 1 = -2