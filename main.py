# import ElFarol_Arthur as E
import ElFarol_GaussianThreshold as EGT

memorias = [1,3,6,9,12]
predictores = [1,3,6,9,12]
num_agentes = [100]
num_rondas = [100]
num_experimentos = 100
umbral = .6
std_umbrales = [0.1, 0.2, 0.4, 0.8]

# E.correr_sweep(memorias, predictores, num_experimentos, num_agentes, umbral, num_rondas, espejos=True, DEB=False)
# E.correr_sweep(memorias, predictores, num_experimentos, num_agentes, umbral, num_rondas, espejos=False, DEB=False)
# print('Barrido finalizado!')

# EGT.simulacion(num_agentes=10, umbral=umbral, std_umbral=0.4, num_predictores=1, long_memoria=2, num_rondas=3, DEB=True, to_file=False)
EGT.correr_sweep(memorias, predictores, num_experimentos, num_agentes, umbral, std_umbrales, num_rondas, espejos=True, DEB=False)
print('Barrido finalizado!')
