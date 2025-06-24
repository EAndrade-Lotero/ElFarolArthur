# import ElFarol_Arthur as E
import ElFarol_GaussianThreshold as EGT

# memorias = [1,3,6,9,12]
# predictores = [1,3,6,9,12]
memorias = [12]
predictores = [12]
num_agentes = [100]
num_rondas = [100]
num_experimentos = 100
umbral = .6
# std_umbrales = [0.1, 0.2, 0.4, 0.8]
# std_umbrales = [0.03, 0.05, 0.06, 0.07, 0.09]
std_umbrales = [0]

# E.correr_sweep(memorias, predictores, num_experimentos, num_agentes, umbral, num_rondas, espejos=True, DEB=False)
# E.correr_sweep(memorias, predictores, num_experimentos, num_agentes, umbral, num_rondas, espejos=False, DEB=False)
# print('Barrido finalizado!')

# EGT.simulacion(num_agentes=100, umbral=umbral, std_umbral=0.08, num_predictores=12, long_memoria=12, num_rondas=100, DEB=False, to_file=True)
EGT.correr_sweep(memorias, predictores, num_experimentos, num_agentes, umbral, std_umbrales, num_rondas, espejos=True, DEB=False)
print('Finalizado!')
