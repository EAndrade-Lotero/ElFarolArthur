import pandas as pd
import numpy as np
from itertools import product

def generar_ejemplo1(id=1):
    '''
    Genera una simulacion en la cual solo un jugador va al bar
    '''
    di = {}
    di['Identificador'] = [id]*20
    di['Ronda'] = list(range(10)) + list(range(10))
    di['Agente'] = [0]*10 + [1]*10
    di['Estado'] =  [0]*10 + [1]*10
    di['Puntaje'] = [0]*10 + [1]*10
    return pd.DataFrame(di)

def generar_ejemplo2(id=1):
    '''
    Genera una simulacion en la cual ambos jugadores se turnan
    '''
    di = {}
    di['Identificador'] = [id]*20
    di['Ronda'] = list(range(10)) + list(range(10))
    di['Agente'] = [0]*10 + [1]*10
    di['Estado'] =  [0,1]*10
    di['Puntaje'] = [0,1]*10
    return pd.DataFrame(di)

def generar_ejemplo3(id=1):
    '''
    Genera una simulacion en la cual ambos jugadores siempre van
    '''
    di = {}
    di['Identificador'] = [id]*20
    di['Ronda'] = list(range(10)) + list(range(10))
    di['Agente'] = [0]*10 + [1]*10
    di['Estado'] =  [1]*20
    di['Puntaje'] = [-1]*20
    return pd.DataFrame(di)

def generar_ejemplo4(id=1):
    '''
    Genera una simulacion en la cual ningún jugador va
    '''
    di = {}
    di['Identificador'] = [id]*20
    di['Ronda'] = list(range(10)) + list(range(10))
    di['Agente'] = [0]*10 + [1]*10
    di['Estado'] =  [0]*20
    di['Puntaje'] = [0]*20
    return pd.DataFrame(di)

def generar_ejemplo5(id=1):
    '''
    Genera una simulacion en la cual se turnan y luego no van
    '''
    di = {}
    di['Identificador'] = [id]*20
    di['Ronda'] = list(range(10)) + list(range(10))
    di['Agente'] = [0]*10 + [1]*10
    di['Estado'] =  [0,1]*2 + [0]*6 + [1,0]*2 + [0]*6
    di['Puntaje'] = [0,1]*2 + [0]*6 + [1,0]*2 + [0]*6
    return pd.DataFrame(di)

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

def leer_datos_aleatorio(probabilidades, conectividades, num_agentes, verb=True):
    names = ['Identificador', 'Ronda', 'Agente', 'Estado', 'Puntaje']
    df_list = []
    for n in num_agentes:
        for d in probabilidades:
            for p in conectividades:
                prob = str(d).replace('.','_0')
                if verb:
                    print(f"Leyendo datos sweep probabilidad {prob}, conectividad {p} y número de agentes {n}")
                archivo = './data/data_aleatorio/simulacion-' + str(prob) + '-' + str(p) + '-' + str(n) + '.csv'
                if verb:
                    print(f"Cargando datos de archivo {archivo}...")
                try:
                    aux = pd.read_csv(archivo, names=names, header=None)
                    aux['Prob'] = d
                    aux['Conectividad'] = p
                    aux['Num_agentes'] = n
                    # print(aux.head())
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
        data['Prob'] = data['Prob'].astype(float)
        data['Conectividad'] = data['Conectividad'].astype(float)
        data['Num_agentes'] = data['Num_agentes'].astype(float)
        data['Identificador'] = data['Identificador'].astype(int)
        data['Ronda'] = data['Ronda'].astype(int)
        data['Agente'] = data['Agente'].astype(int)
        data['Estado'] = data['Estado'].astype(int)
        data['Puntaje'] = data['Puntaje'].astype(int)
    except:
        data = data.iloc[1:]
        if verb:
        	print(data.head())
        # data = data.dropna()
        data['Prob'] = data['Prob'].astype(float)
        data['Conectividad'] = data['Conectividad'].astype(float)
        data['Num_agentes'] = data['Num_agentes'].astype(float)
        data['Identificador'] = data['Identificador'].astype(int)
        data['Ronda'] = data['Ronda'].astype(int)
        data['Agente'] = data['Agente'].astype(int)
        data['Estado'] = data['Estado'].astype(int)
        data['Puntaje'] = data['Puntaje'].astype(int)
    data = data[['Prob', 'Conectividad','Num_agentes', 'Identificador','Ronda','Agente','Estado','Puntaje']]
    if verb:
	    print("Shape:", data.shape)
	    print("Probabilidades value counts:", data['Prob'].value_counts())
	    print("Conectividades value counts:", data['Conectividad'].value_counts())
	    print("Num_agentes value counts:", data['Num_agentes'].value_counts())
	    print("Agente value counts:", data['Agente'].value_counts())
    return data

def encuentra_gap(df):
    '''
    Fairness = (max(rho1',rho2')-min(rho1',rho2'))/max(rho1',rho2')
    donde rhoi' es el número de rondas en las cuales el jugador
    i-ésimo obtiene el mejor puntaje.
    '''
    fairness = []
    for sim, grp_sim in df.groupby('Identificador'):
        rho_sim = []
        for a, grp_a in grp_sim.groupby('Agente'):
            n = grp_a.Puntaje.tolist().count(1)
            rho_sim.append(n)
        m = float(min(rho_sim))
        M = max(rho_sim)
        fairness.append((M-m)/M if M != 0 else 1.)
    return fairness

def encuentra_gap_anterior(df):
    '''
    Fairness = min(rho1',rho2')/max(rho1',rho2')
    donde rhoi' es el número de rondas en las cuales el jugador
    i-ésimo obtiene el mejor puntaje.
    '''
    fairness = []
    for sim, grp_sim in df.groupby('Identificador'):
        rho_sim = []
        for a, grp_a in grp_sim.groupby('Agente'):
            n = grp_a.Puntaje.tolist().count(1)
            rho_sim.append(n)
        m = float(min(rho_sim))
        M = max(rho_sim)
        fairness.append(m/M if M != 0 else 1.)
    return fairness

def encuentra_efficiency(df):
    '''
    Efficiency = rho1 + rho2
    donde rhoi es la ganacia total del jugador i-ésimo
    '''
    return df.groupby('Identificador')['Puntaje'].sum().tolist()

def encuentra_m_efficiency(df):
    '''
    Mean_Efficiency = mean(rho1, rho2)
    donde rhoi es la ganacia promedio del jugador i-ésimo
    '''
    return df.groupby('Identificador')['Puntaje'].mean().tolist()

def encuentra_gini(df):
    '''
    Gini = indice gini
    '''
    gini = []
    for sim, grp_sim in df.groupby('Identificador'):
        grp_sim['Puntaje_N'] = (grp_sim['Puntaje'] + 1) / 2
        x = grp_sim.groupby('Agente').Puntaje_N.sum().tolist()
        # x = [i if i>0 else 0 for i in x]
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(x, x)).mean()
        # Relative mean absolute difference
        rmad = mad/np.mean(x)
        # Gini coefficient
        g = 0.5 * rmad
        gini.append(g)
    return gini

def encuentra_findex(df):
    '''
    Time-Box Fairness Index (TBFI) (Ponsiglione et al., 2015)
    Ajustado para considerar que no todas las rondas se usan todos
    los recursos.
    '''
    df['lleva_recurso'] = df['Puntaje'].apply(lambda x: 1 if x == 1 else 0)
    tbfi = []
    for sim, grp_sim in df.groupby('Identificador'):
        rounds = (grp_sim['Ronda'].max() - grp_sim['Ronda'].min()) + 1
        n_agentes = grp_sim['Agente'].max() + 1
        resour = grp_sim['lleva_recurso'].sum()
        tasa = .6
        assert(resour <= rounds*tasa*n_agentes), f'{resour}, {rounds*tasa*n_agentes}'
        numerator = grp_sim.groupby('Agente')['lleva_recurso'].sum().reset_index(name='xi')
        numerator['num'] = (numerator['xi'] - resour/n_agentes)**2
        numerator = numerator['num'].sum()
        W = np.floor(resour/rounds)
        B = np.ceil(resour/rounds) - np.floor(resour/rounds)
        napa = resour % rounds
        # print(rounds, resour, n_agentes, resour/n_agentes, W, napa)
        denominator = (rounds-resour/n_agentes)**2*W + (napa-resour/n_agentes)**2*B + (resour/n_agentes)**2*(n_agentes-W-B)
        # print(f'({rounds}-{resour/n_agentes})^2*{W} + ({napa}-{resour/n_agentes})^2*{B} + ({resour/n_agentes})^2*{n_agentes-W-B} = {denominator}')
        assert(numerator >= 0)
        assert(denominator >= 0), f'({rounds}-{resour/n_agentes})^2*{W} + ({napa}-{resour/n_agentes})^2*{B} + ({resour/n_agentes})^2*{n_agentes-W-B} = {denominator}'
        if denominator != 0:
            iniq = np.sqrt(numerator/denominator)
        else:
            # print('Atención: denominador 0 en Iniquity Index!')
            iniq = 0
        tbfi.append(1 - iniq)
    return tbfi

def encuentra_findex_anterior(df):
    '''
    Time-Box Fairness index (TFI) (Ponsiglione et al., 2015)
    '''
    n_rondas = (df['Ronda'].max() - df['Ronda'].min()) + 1
    n_agentes = df['Agente'].max() + 1
    FQ = 3
    FP = 5
    a = n_rondas / FP
    tasa = FQ / FP
    MTI = np.sqrt(a**2 * (FQ*FP - FQ**2))
    findex = []
    for sim, grp_sim in df.groupby('Identificador'):
        grp_sim['lleva_recurso'] = grp_sim['Puntaje'].apply(lambda x: 1 if x == 1 else 0)
        TII = np.sqrt(grp_sim.groupby('Agente')['lleva_recurso'].apply(lambda x: (x.sum() - a*FQ)**2).sum()/n_agentes)
        assert(TII <= MTI), f'{sim}, {TII}, {MTI}'
        findex.append(1 - TII/MTI)
    return findex #, TII, MTI

def encuentra_precision(df):
    return df.groupby('Identificador')['Precision'].mean().tolist()

def merge_modelos(df):
    modelos = df.Modelo.unique().tolist()
    df['Concurrencia'] = df.groupby(['Modelo','Identificador','Ronda'])['Estado'].transform('mean')
    m_attendance = df.groupby(['Modelo','Identificador'])['Concurrencia'].mean().reset_index(name='Attendance')
    sd_attendance = df.groupby(['Modelo','Identificador'])['Concurrencia'].std().reset_index(name='Deviation')
    data_s = []
    try:
        a = df['Precision'].unique()
        for mod, grp in df.groupby('Modelo'):
            data_s.append(pd.DataFrame({'Efficiency':encuentra_m_efficiency(grp),'Gap':encuentra_gap(grp), 'Gini':encuentra_gini(grp), 'Inaccuracy':encuentra_precision(grp), 'TBFI':encuentra_findex(grp), 'Identificador':grp['Identificador'].unique().tolist(),'Modelo':mod}))
    except:
        for mod, grp in df.groupby('Modelo'):
            data_s.append(pd.DataFrame({'Efficiency':encuentra_m_efficiency(grp),'Gap':encuentra_gap(grp), 'Gini':encuentra_gini(grp),  'TBFI':encuentra_findex(grp), 'Identificador':grp['Identificador'].unique().tolist(),'Modelo':mod}))

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
    # m_attendance = df1.groupby(parametros+['Identificador'])['Estado'].mean().reset_index(name='Attendance')
    # sd_attendance = df1.groupby(parametros+['Identificador'])['Estado'].std().reset_index(name='Deviation')
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
        if 'Efficiency' in variables:
            diccionario['Efficiency'] = encuentra_m_efficiency(grp)
        if 'Inaccuracy' in variables:
            diccionario['Inaccuracy'] = encuentra_precision(grp)
        if 'Gini' in variables:
            diccionario['Gini'] = encuentra_gini(grp)
        if 'TBFI' in variables:
            diccionario['TBFI'] = encuentra_findex(grp)
        data_s.append(pd.DataFrame(diccionario))
    df_ = pd.concat(data_s)
    df_ = pd.merge(df_, m_attendance, on=parametros+['Identificador'])
    df_ = pd.merge(df_, sd_attendance, on=parametros+['Identificador'])
    print("Dataframe listo!")
    return df_
