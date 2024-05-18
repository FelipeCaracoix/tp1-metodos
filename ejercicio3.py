from matricesRalas import ListaEnlazada, MatrizRala, GaussJordan, multiplicar_matriz_vector
# a = 0
# b = 1
# c = 2
# d = 3
# e = 4
# f = 5
# g = 6
# h = 7
# i = 8
# j = 9
# k = 10

len_letras = 11

w = MatrizRala(len_letras,len_letras)

assignments = {
    (1, 0): 1, (5, 0): 1, (6, 0): 1, # B, F y G citan a A
    (0, 2): 1, #A cita a C
    (0, 3): 1, (0, 4): 1,
    (8, 5): 1,
    (5, 6): 1,
    (6, 7): 1,
    (6, 8): 1, (7, 8): 1, (9, 8): 1,
    (4, 10): 1
}

def build_w(matriz, citas):
    for indices, value in assignments.items():
        w[indices] = value
    return w


W = build_w(w, assignments)
print(W)

def build_d(matriz_w):
    D = MatrizRala(matriz_w.shape[0], matriz_w.shape[1])
    
    for i in range(matriz_w.shape[1]):
        cj = 0
        for j in range(matriz_w.shape[0]):
            cj += matriz_w[j,i]
        if cj != 0:
            D[i,i] = 1 if (round(1/cj, 3) == 1) else round(1/cj , 3)
        else:
            D[i,i] = 0

    return D

def matriz_identidad(n,m):
    matriz = MatrizRala(n,m)

    for i in range(matriz.shape[1]):
        matriz[i,i] = 1
    return matriz

def matriz_de_unos(n,m):
    matriz = MatrizRala(n,m)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            matriz[i,j] = 1
    return matriz


D = build_d(W)

# (identidad - d*W*D)*pestrella = (1-d)/N * 1|
# d = 0.85
matriz_identidad = matriz_identidad(len_letras,len_letras)

d = 0.85
N = len_letras

A = (matriz_identidad - (d*W)@D)
b = MatrizRala(len_letras,1)
for i in range(len_letras):
    b[i,0]= (1 - d)/N

pestrella = GaussJordan(A,b)
print(pestrella)
suma = 0 
for i in range(len_letras):
    suma = suma + pestrella[i,0]
print("la suma de pestrella da: ", suma)


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

pIt, errores = P_it(0.85,len_letras,W,D)
print("\n\nMétodo iterativo de PageRank con distribución inicial equiprobable:", pIt)
print("\n\nErrores en cada iteración:", errores)




def print_ranking(pestrella):
    # Crear una lista de tuplas (índice, valor)
    ranking = [(i, pestrella[i,0]) for i in range((pestrella.shape[0]))]
    
    # Ordenar la lista en orden descendente por el valor
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    # Crear un diccionario para mapear los índices a las letras
    letras = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K'}
    
    # Imprimir el ranking
    for i, valor in ranking:
        print(f'Paper {letras[i]}: {valor}')

# Llamar a la función para imprimir el ranking
print_ranking(pestrella)