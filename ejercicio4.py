import pandas as pd
from matricesRalas import *
import csv

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


def matriz_de_unos(n,m):
    matriz = MatrizRala(n,m)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            matriz[i,j] = 1
    return matriz

def P_it(d,N,W,D):
    p_t = MatrizRala(N,1)     # Initial equiprobable distribution
    for i in range(N):
        p_t[i,0] = 1/N

    tolerance = 1e-6
    errores = []
    error = 1

    mat_unos = matriz_de_unos(N,1)
    unoMenosDeSobreEne = ((1-d)/N) * mat_unos
    d_W = d * W
    d_WD = d_W @ D


    while error > tolerance:
        # Multiplica la matriz W_D por el vector p_t y escala por d
        p_t_plus_1 = d_WD @ p_t
        p_t_plus_1 = unoMenosDeSobreEne + p_t_plus_1
        # Calcula el error máximo en esta iteración comparando el nuevo vector de PageRank con el anterior
        error = max(abs(p_t_plus_1[i,0] - p_t[i,0]) for i in range(N))
        errores.append(error)

        # Actualiza el vector de PageRank para la próxima iteración
        p_t = p_t_plus_1
    return p_t, errores


def main():

    N = len(papers)
    d = 0.85

    page_ranks = P_it(d, N, W, D)

    # Create list of (PageRank score, index)
    lista = [(page_ranks[0][i, 0], i) for i in range(len(papers))]

    # Sort by PageRank score in descending order
    sorted_papers = sorted(lista, key=lambda x: x[0], reverse=True)

    # Print the top 10 papers with podium ranking
    print("Top 10 Papers by PageRank:")
    for rank, (score, paper_id) in enumerate(sorted_papers[:10], start=1):
        paper = papers.loc[paper_id]
        print(f"{rank}. Paper ID: {paper['id']}, Title: \"{paper['titulo']}\", Score: {score}")

if __name__ == "__main__":
    main()