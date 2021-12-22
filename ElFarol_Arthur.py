print("Importando paquetes...")
from random import choice, sample, randint, uniform
import numpy as np
import pandas as pd
import redes
from os import remove
print("Listo!")

def distancia(x, y):
    return abs(x - y)

class Predictor:
    def __init__(self, long_memoria, indicador, espejos):
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
        self.indicador = indicador

    def predecir(self, memoria, num_agentes, umbral):
        long_memoria = len(memoria)
        ciclico = self.ciclico
        ventana = self.ventana
        espejo = self.espejo
        if ventana > 0:
	        if ciclico:
	            # indices = list(range(long_memoria - retardo, -1, -retardo))
	            indices = list(range(long_memoria - 1, -1, -ventana))
	            valores = [memoria[x] for x in indices]
	        else:
	            # valores = historia[max(long_memoria-retardo-ventana+1, 0):max(long_memoria-retardo+1, 0)]
	            valores = memoria[-ventana:]
	        try:
	            prediccion = int(np.mean(valores))
	        except:
	            prediccion = memoria[-1]
	        if espejo:
	            prediccion = num_agentes - prediccion
        else:
	        if uniform(0,1) <= umbral:
	            prediccion = int(umbral*num_agentes - 1) # Decision aleatoria de ir a El Farol con probabilidad umbral
	        else:
	            prediccion = int(umbral*num_agentes + 1) # Decide no ir con probabilidad 1 - umbral
        self.prediccion.append(prediccion)

    def __str__(self):
        ventana = str(self.ventana)
        ciclico = "ciclico" if self.ciclico else "ventana"
        espejo = "-espejo" if self.espejo else ""
        return ventana + "-" + ciclico + espejo + f"({self.indicador})"

class Agente:
    def __init__(self, estados, scores, vecinos, predictores, predictor_activo):
        self.estado = estados # lista
        self.score = scores # lista
        self.vecinos = vecinos # lista
        self.predictores = predictores # lista
        self.predictor_activo = predictor_activo # lista

    def __str__(self):
        return "E:{0}, S:{1}, V:{2}, P:{3}".format(self.estado, self.score, self.vecinos, str(self.predictor_activo[-1]))

class Bar:
    def __init__(self, num_agentes, umbral, long_memoria, num_predictores, conectividad, identificador, espejos):
        self.num_agentes = num_agentes
        self.umbral = umbral
        self.long_memoria = long_memoria
        self.num_predictores = num_predictores
        self.conectividad = conectividad
        self.identificador = identificador
        self.historia = []
        self.predictores = []
        for i in range(100):
            p = Predictor(self.long_memoria,i,espejos)
            self.predictores.append(p)
        self.agentes = []
        for i in range(self.num_agentes):
            predictores_agente = sample(self.predictores, self.num_predictores)
            # print(f"Predictores agente {i}:", str([str(p) for p in predictores_agente]))
            self.agentes.append(Agente([randint(0,1)], [], [], predictores_agente, [choice(predictores_agente)]))
        self.calcular_asistencia() # Encuentra la asistencia al bar
        self.calcular_puntajes() # Encuentra los puntajes de los agentes
        self.actualizar_predicciones() # Predice de acuerdo a la primera asistencia aleatoria
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
            prediccion = a.predictor_activo[-1].prediccion[-1] / self.num_agentes
            if prediccion <= self.umbral:
                a.estado.append(1)
            else:
                a.estado.append(0)

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

    def actualizar_predicciones(self):
        historia = self.historia[-self.long_memoria:]
        # print("Historia para predecir:", historia)
        for p in self.predictores:
            p.predecir(historia, self.num_agentes, self.umbral)

    def actualizar_precision_promedio(self):
        historia = self.historia[-self.long_memoria - 1:]  # por qué el -1 ? Para que historia siempre sea uno más larga que predicciones
        for p in self.predictores:
            if self.long_memoria == 0:
	            p.precision.append(1)
            else:
	            predicciones = p.prediccion[-self.long_memoria:]
	            # print("Historia vs prediccion", historia, predicciones)
	            precision_historia = np.mean([distancia(historia[i + 1], predicciones[i]) for i in range(len(historia) - 1)])
	            p.precision.append(precision_historia)

    def actualizar_precision(self, theta=0.7):
        historia = self.historia
        for p in self.predictores:
            predicciones = p.prediccion
            n = len(predicciones)
            # print("Historia vs prediccion", historia, predicciones)
            # print("len(historia):", len(historia), " len(predicciones): ", len(predicciones), " n: ", n)
            if n == 1:
            	precision_historia = distancia(historia[-1], predicciones[-1])
            else:
            	precision_historia = theta*p.precision[-1] + (1-theta)*(distancia(historia[-1], predicciones[-1]))
            p.precision.append(precision_historia)

    def escoger_predictor(self, DEB=False):
        for a in self.agentes:
            precisiones = [p.precision[-1] for p in a.predictores]
            index_min = np.argmin(precisiones)
            if DEB:
                print("Las precisiones son:")
                print([f"{str(p)} : {p.precision[-1]}" for p in a.predictores])
            a.predictor_activo.append(a.predictores[index_min])

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

    def juega_ronda(self, ronda):
        self.calcular_estados()
        self.calcular_asistencia()
        self.calcular_puntajes()
        self.actualizar_precision()
        self.escoger_predictor(DEB=False)
        self.agentes_aprenden(ronda=ronda, n=5, DEB=False)
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
                estado.append(self.agentes[i].estado[r])
                puntaje.append(self.agentes[i].score[r])
                politica.append(str(self.agentes[i].predictor_activo[r]))
                prediccion.append(self.agentes[i].predictor_activo[r].prediccion[r])
                precision.append(self.agentes[i].predictor_activo[r].precision[r])
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
        data['Identificador'] =  id
        data['Memoria'] = self.long_memoria
        data['Num_predic'] = self.num_predictores
        data['Conectividad'] = self.conectividad
        data = data[['Memoria', 'Num_predic', 'Identificador','Ronda','Agente',\
                     'Estado','Puntaje','Politica','Prediccion', 'Precision']]
        return data

def guardar(dataFrame, archivo, inicial, espejos=True, muchos=False):
    if not muchos:
        if espejos:
        	archivo = "../Data_Farol/normal/data_todo/" + archivo
        else:
        	archivo = "../Data_Farol/normal/data_sin_espejos/" + archivo
    else:
        if espejos:
        	archivo = "../Data_Farol/data_todo/" + archivo
        else:
        	archivo = "../Data_Farol/data_sin_espejos/" + archivo
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

def simulacion(num_agentes, umbral, long_memoria, num_predictores, num_rondas, conectividad, inicial=True, identificador='', espejos=True, DEB=False):
    bar = Bar(num_agentes, umbral, long_memoria, num_predictores, conectividad, identificador, espejos)
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
    archivo = 'simulacion-' + str(long_memoria) + '-' + str(num_predictores) + '-' + str(conectividad) +'.csv'
    if num_agentes < 1000:
        guardar(data, archivo, inicial, espejos)
    else:
        guardar(data, archivo, inicial, espejos, muchos=True)
    # print('Datos guardados en ', archivo)
    # guardar(data, 'agentes.csv', inicial)

def correr_sweep(memorias, predictores, conectividades, num_experimentos, num_agentes, umbral, num_rondas, espejos=True, DEB=False):
    print('********************************')
    print('Corriendo simulaciones...')
    print('********************************')
    print("")
    identificador = 0
    for d in memorias:
        for k in predictores:
            for p in conectividades:
                inicial = True
                print('Corriendo experimentos con parametros:')
                print(f"Memoria={d}; Predictores={k}; Conectividad={p}")
                for i in range(num_experimentos):
                    redes.random_graph(num_agentes, p, imagen=False, identificador=identificador)
                    simulacion(num_agentes, umbral, d, k, num_rondas, p, inicial=inicial, identificador=identificador, espejos=espejos, DEB=DEB)
                    identificador += 1
                    inicial = False
