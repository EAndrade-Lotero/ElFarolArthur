import ElFarol_Arthur as E

memorias = [1, 3, 6, 9, 12]
predictores = [1, 3, 6, 9, 12]
conectividades = [0]
num_experimentos = 100
num_agentes = 1000
umbral = .6
num_rondas = 100

E.correr_sweep(memorias, predictores, conectividades, num_experimentos, num_agentes, umbral, num_rondas, espejos=True, DEB=False)

print('Barrido finalizado!')
