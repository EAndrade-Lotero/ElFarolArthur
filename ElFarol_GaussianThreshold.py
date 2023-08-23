# Implementa una versión en la cual cada agente tiene su propio umbral.
# Los umbrales son generados con base en una distribución gaussiana
# alrededor del umbral del bar.

print("Importando paquetes...")
from random import choice, sample, randint, uniform
import numpy as np
import pandas as pd
from os import remove
from itertools import product
print("Listo!")

# Inicializa generador de números aleatorios
rng = np.random.default_rng()

def distancia(x, y):
    return abs(x - y)

class Predictor:
    def __init__(self, long_memoria, espejos):
        if long_memoria < 1:
            self.ventana = 0
        else:
            self.ventana = randint(1, long_memoria)
        self.ciclico = choice([True, False])
        if espejos:
            self.espejo = choice([True, False])
        else:
            self.espejo = False
        self.precision = [np.nan]
        self.prediccion = []

    def predecir(self, memoria, num_agentes, umbral):
        long_memoria = len(memoria)
        ciclico = self.ciclico
        ventana = self.ventana
        espejo = self.espejo
        if ciclico:
            indices = list(range(long_memoria - 1, -1, -ventana))
            valores = [memoria[x] for x in indices]
        else:
            valores = memoria[-ventana:]
        try:
            prediccion = int(np.mean(valores))
        except:
            prediccion = memoria[-1]
        if espejo:
            prediccion = num_agentes - prediccion
        self.prediccion.append(prediccion)

    def __str__(self):
        ventana = str(self.ventana)
        ciclico = "-ciclico" if self.ciclico else "-ventana"
        espejo = "-espejo" if self.espejo else ""
        return ventana + ciclico + espejo

class Agente:
    def __init__(self, estados, scores, predictores, predictor_activo):
        self.estado = estados # lista
        self.score = scores # lista
        self.predictores = predictores # lista
        self.predictor_activo = predictor_activo # lista

    def __str__(self):
        return "E:{0}, S:{1}, P:{2}".format(self.estado, self.score, str(self.predictor_activo[-1]))

class Bar:
    def __init__(self, num_agentes, umbral, std_umbral, long_memoria, num_predictores, identificador, espejos):
        self.num_agentes = num_agentes
        self.umbral = umbral
        self.std_umbral = std_umbral
        self.long_memoria = long_memoria
        self.num_predictores = num_predictores
        self.identificador = identificador
        self.historia = []
        self.predictores = []
        # Crear todos los predictores posibles
        ventanas = list(range(1,long_memoria+1))
        ciclicos = [True, False]
        if espejos:
            espejos_ = [True, False]
        else:
            espejos_ = [False]
        tuplas = product(ventanas, ciclicos, espejos_)
        for tupla in tuplas:
            p = Predictor(self.long_memoria,espejos)
            p.ventana = tupla[0]
            p.ciclico = tupla[1]
            p.espejo = tupla[2]
            self.predictores.append(p)
        self.agentes = []
        for i in range(self.num_agentes):
            if self.num_predictores <= len(self.predictores):
                predictores_agente = sample(self.predictores, self.num_predictores)
            else:
                predictores_agente = self.predictores
            # print(f"Predictores agente {i}:", str([str(p) for p in predictores_agente]))
            self.agentes.append(Agente([randint(0,1)], [], predictores_agente, [choice(predictores_agente)]))
        # Genera los umbrales de los agentes mediante una distribución normal, con media umbral y desviación std_umbral
        self.umbrales = rng.normal(loc=umbral, scale=std_umbral, size=self.num_agentes) 
        self.calcular_asistencia() # Encuentra la asistencia al bar
        self.calcular_puntajes() # Encuentra los puntajes de los agentes
        self.actualizar_predicciones() # Predice de acuerdo a la primera asistencia aleatoria

    def calcular_estados(self):
        for i, a in enumerate(self.agentes):
            prediccion = a.predictor_activo[-1].prediccion[-1] / self.num_agentes
            if prediccion <= self.umbrales[i]:
                a.estado.append(1)
            else:
                a.estado.append(0)

    def calcular_asistencia(self):
        asistencia = np.sum([a.estado[-1] for a in self.agentes])
        self.historia.append(asistencia)

    def calcular_puntajes(self):
        asistencia = self.historia[-1]/self.num_agentes
        for i, a in enumerate(self.agentes):
            if a.estado[-1] == 1:
                if asistencia > self.umbrales[i]:
                    a.score.append(-1)
                else:
                    a.score.append(1)
            else:
                a.score.append(0)

    def actualizar_predicciones(self):
        historia = self.historia[-self.long_memoria:]
        # print("Historia para predecir:", historia)
        for p in self.predictores:
            p.predecir(historia, self.num_agentes, self.umbral)

    def actualizar_precision(self):
        historia = self.historia
        for p in self.predictores:
            if self.long_memoria == 0:
                p.precision.append(1)
            else:
                predicciones = p.prediccion
                precision_historia = np.mean([distancia(historia[i + 1], predicciones[i]) for i in range(len(historia) - 1)])
                p.precision.append(precision_historia)

    def escoger_predictor(self, DEB=False):
        for a in self.agentes:
            precisiones = [p.precision[-1] for p in a.predictores]
            index_min = np.argmin(precisiones)
            if DEB:
                print("Las precisiones son:")
                print([f"{str(p)} : {p.precision[-1]}" for p in a.predictores])
            a.predictor_activo.append(a.predictores[index_min])

    def juega_ronda(self, ronda):
        self.calcular_estados()
        self.calcular_asistencia()
        self.calcular_puntajes()
        self.actualizar_precision()
        self.escoger_predictor(DEB=False)
        self.actualizar_predicciones()

    def crea_dataframe_agentes(self):
        ronda = []
        agente = []
        estado = []
        puntaje = []
        politica = []
        prediccion = []
        precision = []
        num_iteraciones = len(self.historia) - 1
        for i in range(self.num_agentes):
            for r in range(num_iteraciones):
                agente.append(i)
                ronda.append(r)
                a = self.agentes[i]
                p = a.predictor_activo[r]
                estado.append(a.estado[r])
                puntaje.append(a.score[r])
                politica.append(str(p))
                prediccion.append(p.prediccion[r])
                precision.append(p.precision[r])
        data = pd.DataFrame.from_dict(\
                                    {\
                                    'Ronda': ronda,\
                                    'Agente': agente,\
                                    'Estado': estado,\
                                    'Puntaje': puntaje,\
                                    'Politica': politica,\
                                    'Precision': precision,\
                                    'Prediccion': prediccion\
                                    })

        id = self.identificador if self.identificador != '' else 'A'
        data['Identificador'] = id
        data['Memoria'] = self.long_memoria
        data['Num_predic'] = self.num_predictores
        data['Std_umbral'] = self.std_umbral
        data['Num_agentes'] = self.num_agentes
        data = data[['Memoria', 'Num_predic', 'Std_umbral', 'Identificador','Ronda','Agente',\
                     'Estado','Puntaje','Politica','Prediccion', 'Precision']]
        return data

def guardar(dataFrame, archivo, inicial, espejos=True, muchos=False):
    if not muchos:
        if espejos:
            archivo = "./Data_Farol_Gaussian_Threshold/normal/data_todo/" + archivo
        else:
            archivo = "./Data_Farol_Gaussian_Threshold/normal/data_sin_espejos/" + archivo
    else:
        if espejos:
            archivo = "./Data_Farol_Gaussian_Threshold/data_todo/" + archivo
        else:
            archivo = "./Data_Farol_Gaussian_Threshold/data_sin_espejos/" + archivo
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

def simulacion(num_agentes, umbral, std_umbral, long_memoria, num_predictores, num_rondas, inicial=True, identificador='', espejos=True, DEB=False, to_file=True):
    bar = Bar(num_agentes, umbral, std_umbral, long_memoria, num_predictores, identificador, espejos)
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
    data['Num_rondas'] = num_rondas
    archivo = 'simulacion-' + str(long_memoria) + '-' + str(num_predictores) + '-' + str(std_umbral) + '-' + str(num_agentes) + '-' + str(num_rondas) + '.csv'
    if to_file:
        if num_agentes < 1000:
            guardar(data, archivo, inicial, espejos)
        else:
            guardar(data, archivo, inicial, espejos, muchos=True)
        if DEB:
            print('Datos guardados en ', archivo)
    return bar

def correr_sweep(memorias, predictores, num_experimentos, num_agentes, umbral, std_umbrales, num_rondas, espejos=True, DEB=False):
    print('********************************')
    print('Corriendo simulaciones...')
    print('********************************')
    print("")
    identificador = 0
    for d in memorias:
        for k in predictores:
            for s in std_umbrales:
                for N in num_agentes:
                    for T in num_rondas:
                        inicial = True
                        print('Corriendo experimentos con parametros:')
                        print(f"Memoria={d}; Predictores={k}; Dev Umbral={s}; Número de agentes={N}; Número de rondas={T}; Espejos?:{espejos}")
                        for i in range(num_experimentos):
                            simulacion(N, umbral, s, d, k, T, inicial=inicial, identificador=identificador, espejos=espejos, DEB=DEB)
                            identificador += 1
                            inicial = False
