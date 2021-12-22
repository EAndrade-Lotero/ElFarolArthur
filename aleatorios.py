print("Importando paquetes...")
from random import choice, sample, randint, uniform
import numpy as np
import pandas as pd
import redes
from os import remove
print("Listo!")

class Agente:
    def __init__(self, estados, scores, vecinos, probabilidad):
        self.estado = estados # lista
        self.score = scores # lista
        self.vecinos = vecinos # lista
        self.probabilidad = probabilidad

    def __str__(self):
        return "E:{0}, S:{1}, V:{2}, P:{3}".format(self.estado, self.score, self.vecinos)

class Bar:
    def __init__(self, num_agentes, umbral, probabilidad, conectividad, identificador):
        self.num_agentes = num_agentes
        self.umbral = umbral
        self.conectividad = conectividad
        self.identificador = identificador
        self.historia = []
        self.agentes = []
        for i in range(self.num_agentes):
            estado = 1 if uniform(0,1) <= probabilidad else 0
            self.agentes.append(Agente([estado], [], [], probabilidad))
        self.calcular_asistencia() # Encuentra la asistencia al bar
        self.calcular_puntajes() # Encuentra los puntajes de los agentes
        self.leer_red() # Lee red desde archivo para incluir vecinos

    def leer_red(self):
        net = {}
        aux = '-' if self.identificador != '' else ''
        In = open("data/redes/connlist" + aux + str(self.identificador) + ".dat", "r")
        for line in In:
            v = list(map(int, line.split()))
            if v[0] not in net.keys():
                net[v[0]] = [v[1]]
            else:
                net[v[0]].append(v[1])
            if v[1] not in net.keys():
                net[v[1]] = [v[0]]
            else:
                net[v[1]].append(v[0])
        In.close()
        # print('Red', net)
        for i in range(len(self.agentes)):
            try:
                self.agentes[i].vecinos = net[i]
            except:
                self.agentes[i].vecinos = []

    def calcular_estados(self):
        for a in self.agentes:
            estado = 1 if uniform(0,1) <= a.probabilidad else 0
            a.estado.append(estado)

    def calcular_asistencia(self):
        asistencia = np.sum([a.estado[-1] for a in self.agentes])
        self.historia.append(asistencia)

    def calcular_puntajes(self):
        asistencia = self.historia[-1]/self.num_agentes
        for a in self.agentes:
            if a.estado[-1] == 1:
                if asistencia > self.umbral:
                    a.score.append(-1)
                else:
                    a.score.append(1)
            else:
                a.score.append(0)

    def juega_ronda(self, ronda):
        self.calcular_estados()
        self.calcular_asistencia()
        self.calcular_puntajes()

    def crea_dataframe_agentes(self):
        ronda = []
        agente = []
        estado = []
        puntaje = []
        num_iteraciones = len(self.historia) - 1
        for i in range(self.num_agentes):
            for r in range(num_iteraciones):
                agente.append(i)
                ronda.append(r)
                estado.append(self.agentes[i].estado[r])
                puntaje.append(self.agentes[i].score[r])
        data = pd.DataFrame.from_dict(\
                                    {\
                                    'Ronda': ronda,\
                                    'Agente': agente,\
                                    'Estado': estado,\
                                    'Puntaje': puntaje
                                    })

        id = self.identificador if self.identificador != '' else 'A'
        data['Identificador'] =  id
        data['Conectividad'] = self.conectividad
        data = data[['Identificador','Ronda','Agente',\
                     'Estado','Puntaje']]
        return data

'''
HAY QUE ARREGLAR ESTE CODIGO PARA COPIAR LA PROBABILIDAD DEL VECINO CON MAYOR PUNTAJE
    def copiar_a_vecinos(self, agente, DEB=False):
        # Obtener atributos agente para despues copiar
        estados = [e for e in agente.estado]
        scores = [s for s in agente.score]
        vecinos = [v for v in agente.vecinos]
        predictores = [p for p in agente.predictores]
        predictor_activo = [p for p in agente.predictor_activo]
        # Datos para buscar mejor predictor en vecinos
        predictor = agente.predictor_activo[-1]
        minimo = predictor.precision[-1]
        minimo_vecino = self.agentes.index(agente)
        precisiones_vecinos = [self.agentes[index_vecino].predictor_activo[-1].precision[-1] for index_vecino in agente.vecinos]
        if len(precisiones_vecinos) > 0:
            if DEB:
                print("Considerando agente", minimo_vecino)
                print(agente)
                print("Precision de agente:", minimo, end = "")
                print(" Precision de los vecinos:", precisiones_vecinos)
            if min(precisiones_vecinos) < minimo:
                # Eliminar peor predictor del agente
                precisiones = [p.precision[-1] for p in agente.predictores]
                index_max = np.argmax(precisiones)
                if DEB:
                    print("Precisiones del agente:", precisiones, "La peor es", index_max)
                del predictores[index_max]
                # Añadir mejor predictor de los vecinos
                minimo_vecino = agente.vecinos[np.argmin(precisiones_vecinos)]
                predictor = self.agentes[minimo_vecino].predictor_activo[-1]
                predictores.append(predictor)
                if DEB:
                    print('Se imita el predictor', predictor, 'del vecino', minimo_vecino)
            else:
                if DEB:
                    print('Agente', minimo_vecino, 'no tiene vecinos con mejor precision.')
        else:
            if DEB:
                print('Agente', minimo_vecino, 'no tiene vecinos.')
        predictor_activo[-1] = predictor
        return Agente(estados, scores, vecinos, predictores, predictor_activo)

    def agentes_aprenden(self, ronda=0, n=0, DEB=False):
        # Dejamos n rondas para probar la política escogida
        # En otras palabras, no hay aprendizaje por n rondas.
        # Los agentes copian la politica del vecino con mayor
        # puntaje acumulado en las n rondas. Si n<2, se aprende cada ronda.
        if (n < 2) or (ronda % n == 0):
            Agentes = []
            for agente in self.agentes:
                agente_dummy = self.copiar_a_vecinos(agente, DEB=DEB)
                Agentes.append(agente_dummy)
            self.agentes = Agentes
        else:
            if DEB:
                print("Esta ronda los agentes no aprenden.")
'''

def guardar(dataFrame, archivo, inicial):
    archivo = archivo.replace('.', '_0')
    archivo = "./data/data_aleatorio/" + archivo +'.csv'
    if inicial:
        try:
        	remove(archivo)
        except:
        	pass
        with open(archivo, 'w') as f:
            dataFrame.to_csv(f, header=False, index=False)
    else:
        with open(archivo, 'a') as f:
            dataFrame.to_csv(f, header=False, index=False)

def simulacion(num_agentes, umbral, num_rondas, probabilidad, conectividad, inicial=True, identificador='', DEB=False):
    bar = Bar(num_agentes, umbral, probabilidad, conectividad, identificador)
    if DEB:
        print("**********************************")
        print("Agentes iniciales:")
        for a in bar.agentes:
            print(a)
        print("**********************************")
        print("")
    for i in range(num_rondas):
        if DEB:
            print("Ronda", i)
            print("Historia:", bar.historia)
            # for p in bar.predictores:
            #     print(f"Predictor: {str(p)} - Prediccion: {p.prediccion} - Precision: {p.precision}")
            # print("****************************")
        bar.juega_ronda(i + 1)
        if DEB:
            for a in bar.agentes:
                print(a)
    data = bar.crea_dataframe_agentes()
    archivo = 'simulacion-' + str(probabilidad) + '-' + str(conectividad) + '-' + str(num_agentes)
    guardar(data, archivo, inicial)
    # print('Datos guardados en ', archivo)
    # guardar(data, 'agentes.csv', inicial)

def correr_sweep(probabilidades, conectividades, num_experimentos, num_agentes, umbral, num_rondas, DEB=False):
    print('********************************')
    print('Corriendo simulaciones...')
    print('********************************')
    print("")
    identificador = 0
    for n in num_agentes:
        for d in probabilidades:
            for p in conectividades:
                inicial = True
                print('Corriendo experimentos con parametros:')
                print(f"Probabilidad={d}; Conectividad={p}; Num_agentes={n}")
                for i in range(num_experimentos):
                    redes.random_graph(n, p, imagen=False, identificador=identificador)
                    simulacion(n, umbral, num_rondas, d, p, inicial=inicial, identificador=identificador, DEB=DEB)
                    identificador += 1
                    inicial = False

# probabilidades = [.4,.45,.5,.55,.6,.65]
probabilidades = [0]
conectividades = [0]
num_experimentos = 100
num_agentes = [100, 500, 1000, 2000]
umbral = .6
num_rondas = 100
correr_sweep(probabilidades, conectividades, num_experimentos, num_agentes, umbral, num_rondas)
