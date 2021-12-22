import Procedatos as P
import Graficos as G

print('Leyendo datos...')
data = P.leer_datos([1,3,6,9,12],[1,3,6,9,12],[0],verb=False,muchos=True)
print('Datos leídos!')
variables = ['Deviation']
print('Graficando heatmaps...')
G.graficar_heatmaps(data, parametros=['Memoria','Num_predic'], variables=variables,to_file=True)
print('Listo!')
