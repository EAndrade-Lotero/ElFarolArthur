import ElFarol_Arthur as E

memorias = [1,3,6,9,12]
predictores = [1,3,6,9,12]
num_agentes = [100]
num_rondas = [100]
num_experimentos = 100
umbral = .6

E.correr_sweep(memorias, predictores, num_experimentos, num_agentes, umbral, num_rondas, espejos=True, DEB=False)

E.correr_sweep(memorias, predictores, num_experimentos, num_agentes, umbral, num_rondas, espejos=False, DEB=False)

print('Barrido finalizado!')
