import pandas as pd
import numpy as np
from itertools import product

def leer_datos(memorias, predictores, num_agentes, num_rondas, espejos=True, verb=True, muchos=False, cola=False):
    names = ['Memoria', 'Num_predic', 'Identificador', 'Ronda', 'Agente', 'Estado', 'Puntaje', 'Politica', 'Prediccion', 'Precision', 'Num_agentes']
    df_list = []
    for d in memorias:
        for k in predictores:
            for N in num_agentes:
                for T in num_rondas:
                    if verb:
                        print(f"Leyendo datos sweep memoria {d} predictores {k} número de agentes {N} y número de rondas {T}")
                    if not muchos:
                        if espejos:
                            archivo = '../Data_Farol/normal/data_todo/simulacion-' + str(d) + "-" + str(k) + '-' + str(N) + '-' + str(T) + ".csv"
                        else:
                            archivo = '../Data_Farol/normal/data_sin_espejos/simulacion-' + str(d) + "-" + str(k) + '-' + str(N) + '-' + str(T) + ".csv"
                    else:
                        if espejos:
                            archivo = '../Data_Farol/data_todo/simulacion-' + str(d) + "-" + str(k) + '-' + str(N) + '-' + str(T) + ".csv"
                        else:
                            archivo = '../Data_Farol/data_sin_espejos/simulacion-' + str(d) + "-" + str(k) + '-' + '-' + str(N) + '-' + str(T) + ".csv"
                    if verb:
    	                print(f"Cargando datos de archivo {archivo}...")
                    try:
                        aux = pd.read_csv(archivo, names=names, header=None)
                        if 'Memoria' in aux['Memoria'].unique().tolist():
                            aux = aux.iloc[1:]
                        # aux['Num_agentes'] = N
                        aux['Num_rondas'] = T
                        # print(aux.head())
                        if cola:
                            aux = pd.DataFrame(aux[aux.Ronda>int(max(aux.Ronda)*.8)])
                        df_list.append(aux)
                        if verb:
    	                    print("Listo")
                    except:
                        print(f"Archivo {archivo} no existe! Saltando a siguiente opción")
    if verb:
    	print("Preparando dataframe...")
    data = pd.concat(df_list)
    if verb:
    	print(data.head())
    try:
        # data = data.dropna()
        data['Memoria'] = data['Memoria'].astype(int)
        data['Num_predic'] = data['Num_predic'].astype(int)
        data['Num_agentes'] = data['Num_agentes'].astype(int)
        data['Num_rondas'] = data['Num_rondas'].astype(int)
        data['Identificador'] = data['Identificador'].astype(int)
        data['Ronda'] = data['Ronda'].astype(int)
        data['Agente'] = data['Agente'].astype(int)
        data['Estado'] = data['Estado'].astype(int)
        data['Puntaje'] = data['Puntaje'].astype(int)
        data['Politica'] = data['Politica'].astype(str)
        data['Prediccion'] = data['Prediccion'].astype(int)
        data['Precision'] = data['Precision'].astype(float)
    except:
        data = data.iloc[1:]
        if verb:
        	print(data.head())
        # data = data.dropna()
        data['Memoria'] = data['Memoria'].astype(int)
        data['Num_predic'] = data['Num_predic'].astype(int)
        data['Num_agentes'] = data['Num_agentes'].astype(int)
        data['Num_rondas'] = data['Num_rondas'].astype(int)
        data['Identificador'] = data['Identificador'].astype(int)
        data['Ronda'] = data['Ronda'].astype(int)
        data['Agente'] = data['Agente'].astype(int)
        data['Estado'] = data['Estado'].astype(int)
        data['Puntaje'] = data['Puntaje'].astype(int)
        data['Politica'] = data['Politica'].astype(str)
        data['Prediccion'] = data['Prediccion'].astype(int)
        data['Precision'] = data['Precision'].astype(float)
    data = data[['Memoria','Num_predic','Num_agentes','Num_rondas','Identificador','Ronda','Agente','Estado','Puntaje','Politica', 'Prediccion', 'Precision']]
    if verb:
	    print("Shape:", data.shape)
	    print("Memoria value counts:", data['Memoria'].value_counts())
	    print("Predictores value counts:", data['Num_predic'].value_counts())
	    print("Agente value counts:", data['Num_agentes'].value_counts())
	    print("Rounds value counts:", data['Num_rondas'].value_counts())
    return data

def merge_modelos(df):
    modelos = df.Modelo.unique().tolist()
    df['Concurrencia'] = df.groupby(['Modelo','Identificador','Ronda'])['Estado'].transform('mean')
    m_attendance = df.groupby(['Modelo','Identificador'])['Concurrencia'].mean().reset_index(name='Attendance')
    sd_attendance = df.groupby(['Modelo','Identificador'])['Concurrencia'].std().reset_index(name='Deviation')
    data_s = []
    try:
        a = df['Precision'].unique()
        for mod, grp in df.groupby('Modelo'):
            data_s.append(pd.DataFrame({'Efficiency':grp.groupby('Identificador')['Puntaje'].mean().tolist(),\
            'Inaccuracy':grp.groupby('Identificador')['Precision'].mean().tolist(),\
             'Identificador':grp['Identificador'].unique().tolist(), 'Modelo':mod}))
    except:
        for mod, grp in df.groupby('Modelo'):
            data_s.append(pd.DataFrame({'Efficiency':grp.groupby('Identificador')['Puntaje'].mean().tolist(),\
             'Identificador':grp['Identificador'].unique().tolist(), 'Modelo':mod}))

    df2 = pd.concat(data_s)
    df2 = pd.merge(df2, m_attendance, on=['Modelo','Identificador'])
    df2 = pd.merge(df2, sd_attendance, on=['Modelo','Identificador'])
    return df2

def merge_parametros(df, parametros, variables):
    assert(len(parametros) == 2)
    if ('Attendance' in variables) or ('Deviation' in variables):
        df['Concurrencia'] = df.groupby(parametros+['Identificador','Ronda'])['Estado'].transform('mean')
        m_attendance = df.groupby(parametros+['Identificador'])['Concurrencia'].mean().reset_index(name='Attendance')
        sd_attendance = df.groupby(parametros+['Identificador'])['Concurrencia'].std().reset_index(name='Deviation')
    print("Attendance y Deviation listos!")
    data_s = []
    A = df.groupby(parametros)
    p1 = df[parametros[0]].unique().tolist()
    p2 = df[parametros[1]].unique().tolist()
    for m in product(p1,p2):
        grp = A.get_group(m)
        diccionario = {}
        diccionario[parametros[0]] = m[0]
        diccionario[parametros[1]] = m[1]
        diccionario['Identificador'] = grp['Identificador'].unique().tolist()
        diccionario['Efficiency'] = grp.groupby('Identificador')['Puntaje'].mean().tolist()
        diccionario['Inaccuracy'] = grp.groupby('Identificador')['Precision'].mean().tolist()
        data_s.append(pd.DataFrame(diccionario))
    df_ = pd.concat(data_s)
    df_ = pd.merge(df_, m_attendance, on=parametros+['Identificador'])
    df_ = pd.merge(df_, sd_attendance, on=parametros+['Identificador'])
    print("Dataframe listo!")
    return df_
