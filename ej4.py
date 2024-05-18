import pandas as pd
from matricesRalas import *

papers = pd.read_csv('papers/papers.csv', header = 0)
citas = pd.read_csv('papers/citas.csv', header = 0)

W = MatrizRala(len(papers), len(papers))
D = MatrizRala(len(papers), len(papers))

cantidad_citas = [0] * len(papers)
for index, row in citas.iterrows():
    W[row['to'], row['from']] = 1
    cantidad_citas[row['from']] += 1

for i in range(len(cantidad_citas)):
    if cantidad_citas[i] != 0:
        D[i, i] = 1/cantidad_citas[i]
        
p0 = MatrizRala(len(papers), 1)
for i in range(len(papers)):
    p0[i, 0] = 1/len(papers)

p = [p0]


b = MatrizRala(len(papers), 1)
vector_unos = MatrizRala(len(papers), 1)

for i in range(len(papers)):
    vector_unos[i, 0] = 1
    
for i in range(len(papers)):
    b[i, 0] = ((1 - 0.85)/len(papers)) * vector_unos[i, 0]

dif = 1
epsilon = 1e-6

while dif > epsilon:
    dif = 0
    p.append(b + (0.85 * ((W@D) @ p[-1])))
    for i in range(len(papers)):
        dif += abs(p[-1][i, 0] - p[-2][i, 0])
    print(dif)


p_list = []
suma=0
for i in range(len(papers)):
    p_list.append((p[-1][i,0], i)) #quiero el ultimo que es donde converge
    suma+=p[-1][i,0]

p_sorted = sorted(p_list, key=lambda x: x[0], reverse=True)

mejores_diez = p_sorted[:10]
print(mejores_diez)
print(suma)