print("Importando paquetes...")
from random import choice, sample, randint
import numpy as np
print("Listo!")

class Predictor:
    def __init__(self, long_memoria):
        self.ventana = randint(1, long_memoria)
        self.ciclico = choice([True, False])
        self.espejo = choice([True, False])
        self.precision = [np.nan]
        self.prediccion = []

    def predecir(self, memoria, num_agentes):
        long_memoria = len(memoria)
        ciclico = self.ciclico
        ventana = self.ventana
        espejo = self.espejo
        if ciclico:
            # indices = list(range(long_memoria - retardo, -1, -retardo))
            indices = list(range(long_memoria - 1, -1, -ventana))
            valores = [memoria[x] for x in indices]
        else:
            # valores = historia[max(long_memoria-retardo-ventana+1, 0):max(long_memoria-retardo+1, 0)]
            valores = memoria[-ventana:]
        print("valores:", valores)
        try:
            prediccion = int(np.mean(valores))
        except:
            prediccion = memoria[-1]
        if espejo:
            prediccion = num_agentes - prediccion
        self.prediccion.append(prediccion)

    def __str__(self):
        ventana = str(self.ventana)
        ciclico = "ciclico" if self.ciclico else "ventana"
        espejo = "-espejo" if self.espejo else ""
        return ventana + "-" + ciclico + espejo

class Agente:
    def __init__(self, estados, scores, vecinos, predictores, predictor_activo):
        self.estado = estados # lista
        self.score = scores # lista
        self.vecinos = vecinos # lista
        self.predictores = predictores # lista
        self.predictor_activo = predictor_activo # lista

    def __str__(self):
        return "E:{0}, S:{1}, V:{2}, P:{3}".format(self.estado, self.score, self.vecinos, str(self.predictor_activo[-1]))

historia = [34, 42, 28, 54, 16, 74]
print("Historia:", historia)
a = Predictor(3)
a.ventana = 2
a.ciclico = False
a.espejo = False
print(a)
a.predecir(historia, 100)
print("Prediccion:", a.prediccion)
a = Predictor(3)
a.ventana = 2
a.ciclico = True
a.espejo = False
print(a)
a.predecir(historia, 100)
print("Prediccion:", a.prediccion)