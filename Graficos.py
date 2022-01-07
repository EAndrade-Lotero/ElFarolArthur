import pandas as pd
import numpy as np
import Procedatos as P
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

def graficar_desemp(df):
    fig3 = plt.figure(figsize=(12,6), constrained_layout=True)
    gs = fig3.add_gridspec(2, 4)
    f3_ax1 = fig3.add_subplot(gs[0, :2])
    f3_ax2 = fig3.add_subplot(gs[0, 2])
    f3_ax3 = fig3.add_subplot(gs[0, 3])
    f3_ax4 = fig3.add_subplot(gs[1, 0])
    f3_ax5 = fig3.add_subplot(gs[1, 1])
    f3_ax6 = fig3.add_subplot(gs[1, 2])
    f3_ax7 = fig3.add_subplot(gs[1, 3])
    # Plot attendance
    attendance = df.groupby(['Identificador','Ronda'])['Estado'].mean().reset_index()
    attendance['Concurrencia'] = attendance['Estado']*100
    sns.scatterplot(x='Ronda', y='Concurrencia', data=attendance, ax=f3_ax1, alpha=0.3, marker='+')
    f3_ax1.set_ylabel('Attendance')
    f3_ax1.set_xlabel('Round')
    f3_ax1.set_title('Attendance per round')
    # Plot dist. of attendance
    sns.kdeplot(x='Concurrencia',data=attendance,fill=True,ax=f3_ax2)
    f3_ax2.set_xlabel('Aattendance')
    f3_ax2.set_title('Dist. of attendance')
    # Plot dist. going to bar
    valores = df.groupby(['Identificador','Agente'])['Estado'].sum().tolist()
    sns.kdeplot(valores,fill=True,ax=f3_ax3)
    f3_ax3.set_xlabel('# of rounds agent goes to bar')
    f3_ax3.set_title('Agent goes to bar')
    # Plot dist. of accumulated score per player
    valores = df.groupby(['Identificador','Agente'])['Puntaje'].sum().tolist()
    sns.kdeplot(valores,fill=True,ax=f3_ax4)
    f3_ax4.set_xlabel('Agent\'s accumulated score\n(last 20 rounds)')
    f3_ax4.set_title('Dist. of score')
    # Find measures
    data = pd.DataFrame({'Efficiency':P.encuentra_m_efficiency(df),'Gap':P.encuentra_gap(df), 'Gini':P.encuentra_gini(df), 'Iniquity':P.encuentra_findex(df), 'Identificador':df['Identificador'].unique().tolist()})
    # Plot Gini
    sns.kdeplot('Gini',data=data,fill=True,ax=f3_ax5)
    f3_ax5.set_xlabel('Gini coefficient\n(last 20 rounds)')
    f3_ax5.set_title('Gini')
    # Plot Inequity
    sns.kdeplot('Iniquity',data=data,fill=True,ax=f3_ax6)
    f3_ax6.set_xlabel('Iniquity Index\n(last 20 rounds)')
    f3_ax6.set_title('Iniquity')
    # Plot efficiency
    efficiency = P.encuentra_m_efficiency(df)
    sns.kdeplot('Efficiency',data=data,fill=True,ax=f3_ax7)
    f3_ax7.set_xlabel('Av. score\n(last 20 rounds)')
    f3_ax7.set_title('Efficiency')

def comparar_desemp(df):
    data = P.merge_modelos(df)
    # df1 = pd.DataFrame(df[df['Ronda']>int(max(df.Ronda)*.8)])
    n_agentes = df.Agente.max() + 1
    paleta = "tab10" #'flare'
    modelos = df.Modelo.unique().tolist()
    fig3 = plt.figure(figsize=(12,6), constrained_layout=True)
    gs = fig3.add_gridspec(2, 4)
    f3_ax1 = fig3.add_subplot(gs[0, :2])
    f3_ax2 = fig3.add_subplot(gs[0, 2])
    f3_ax3 = fig3.add_subplot(gs[0, 3])
    f3_ax4 = fig3.add_subplot(gs[1, 0])
    f3_ax5 = fig3.add_subplot(gs[1, 1])
    f3_ax6 = fig3.add_subplot(gs[1, 2])
    f3_ax7 = fig3.add_subplot(gs[1, 3])
    ax = [f3_ax1,f3_ax2,f3_ax3,f3_ax4,f3_ax5,f3_ax6,f3_ax7]
    # Plot attendance per round
    attendance = df.groupby(['Modelo', 'Identificador', 'Ronda'])['Estado'].mean().reset_index()
    attendance['Concurrencia'] = attendance['Estado']*100
    sns.scatterplot(x='Ronda', y='Concurrencia', data=attendance, ax=f3_ax1, alpha=0.3, hue='Modelo', marker='+', palette=paleta)
    f3_ax1.set_ylabel('Attendance')
    f3_ax1.set_xlabel('Round')
    f3_ax1.set_title('Attendance per round')
    # Plot dist. of attendance
    attendance = df.groupby(['Modelo', 'Identificador', 'Ronda'])['Estado'].mean().reset_index()
    attendance['Concurrencia'] = attendance['Estado']*100
    sns.kdeplot(x='Concurrencia',data=attendance,fill=True,ax=f3_ax2, hue='Modelo', palette=paleta)
    f3_ax2.set_xlabel('Aattendance\n(last 20 rounds)')
    f3_ax2.set_title('Dist. of attendance')
    # Plot dist. going to bar
    valores = df.groupby(['Modelo', 'Identificador', 'Agente'])['Estado'].sum().reset_index()
    if valores.Estado.min() < valores.Estado.max():
        sns.kdeplot(x='Estado', data=valores, fill=True,ax=f3_ax3, hue='Modelo', palette=paleta)
    else:
        sns.histplot(x='Estado', data=valores, fill=True, bins=18, multiple='dodge',ax=f3_ax3, hue='Modelo', palette=paleta)
    f3_ax3.set_xlabel('# of rounds agent goes to bar\n(last 20 rounds)')
    f3_ax3.set_title('Agent goes to bar')
    # Plot dist. of mean score per player
    valores = df.groupby(['Modelo', 'Identificador', 'Agente'])['Puntaje'].mean().reset_index()
    if valores.Puntaje.min() < valores.Puntaje.max():
        sns.kdeplot(x='Puntaje',data=valores,fill=True,ax=f3_ax4, hue='Modelo', palette=paleta)
    else:
        sns.histplot(x='Puntaje',data=valores,fill=True, bins=18, multiple='dodge', hue='Modelo', palette=paleta)
    f3_ax4.set_xlabel('Agent\'s average score\n(last 20 rounds)')
    f3_ax4.set_title('Dist. of av. score')
    # Plot Gini
    sns.kdeplot('Gini', data=data,fill=True,ax=f3_ax5,hue='Modelo',palette=paleta)
    f3_ax5.set_xlabel('Gini coefficient\n(last 20 rounds)')
    f3_ax5.set_title('Gini')
    # Plot TBFI
    sns.kdeplot('TBFI', data=data,fill=True,ax=f3_ax6,hue='Modelo',palette=paleta)
    f3_ax6.set_xlabel('TBFI\n(last 20 rounds)')
    f3_ax6.set_title('TBFI')
    # Plot efficiency
    sns.kdeplot('Efficiency', data=data,fill=True,ax=f3_ax7,hue='Modelo',palette=paleta)
    f3_ax7.set_xlabel('Av. score per simulation\n(last 20 rounds)')
    f3_ax7.set_title('Efficiency')

    # # Create the legend
    # f3_ax1.get_legend().remove()
    # f3_ax2.get_legend().remove()
    # f3_ax3.get_legend().remove()
    # f3_ax4.get_legend().remove()
    # f3_ax5.get_legend().remove()
    # f3_ax6.get_legend().remove()
    # f3_ax7.get_legend().remove()
    # fig3.legend(ax,     # The line objects
    #        labels=modelos[::-1],   # The labels for each line
    #        loc='upper center',   # Position of legend
    #        fancybox=True,
    #        shadow=True,
    #        ncol=2,
    #        borderaxespad=0.1,    # Small spacing around legend box
    #        title="Model"  # Title for the legend
    #        )
    #
    # plt.subplots_adjust(top=1.85)

def comparar_desemp_anterior(df):
    paleta = 'flare'
    modelos = df.Modelo.unique().tolist()
    fig, ax = plt.subplots(2,3, figsize=(9,6), tight_layout=True)
    # Plot attendance
    attendance = df.groupby(['Modelo','Identificador','Ronda'])['Estado'].mean().reset_index()
    sns.lineplot(x='Ronda', y='Estado', data=attendance,ax=ax[0,0],hue='Modelo',palette=paleta,)
    ax[0,0].set_ylabel('Attendance (%)')
    ax[0,0].set_xlabel('Round')
    ax[0,0].set_title('Attendance per round')
    ax[0,0].get_legend().remove()
    # Plot dist. going to bar
    dist_go_to = []
    for mod, grp in df.groupby('Modelo'):
        aux = grp.groupby(['Identificador','Agente'])['Estado'].sum().tolist()
        dist_go_to.append(pd.DataFrame({'Dist':aux,'Modelo':mod}))
    df_dist = pd.concat(dist_go_to)
    sns.kdeplot('Dist',data=df_dist,fill=True,ax=ax[0,1],hue='Modelo',palette=paleta)
    ax[0,1].set_xlabel('# of rounds agent goes to bar')
    ax[0,1].set_title('Dist. of going to bar')
    ax[0,1].get_legend().remove()
    # Plot fairness
    fairness = []
    for mod, grp in df.groupby('Modelo'):
        fairness.append(pd.DataFrame({'Gap':P.encuentra_gap(grp),'Modelo':mod}))
    df_fairness = pd.concat(fairness)
    sns.kdeplot('Gap', data=df_fairness,fill=True,ax=ax[0,2],hue='Modelo',palette=paleta)
    ax[0,2].set_xlabel('Gap')
    ax[0,2].set_title('Distribution of fairness')
    ax[0,2].get_legend().remove()
    # Plot Gini
    gini = []
    for mod, grp in df.groupby('Modelo'):
        gini.append(pd.DataFrame({'Gini':P.encuentra_gini(grp),'Modelo':mod}))
    df_gini = pd.concat(gini)
    sns.kdeplot('Gini', data=df_gini,fill=True,ax=ax[1,0],hue='Modelo',palette=paleta)
    ax[1,0].set_xlabel('Gini coefficient')
    ax[1,0].set_title('Distribution of Gini')
    ax[1,0].get_legend().remove()
    # Plot dist. av. score
    dist_go_to = []
    for mod, grp in df.groupby('Modelo'):
        aux = grp.groupby(['Identificador','Agente'])['Puntaje'].sum().tolist()
        dist_go_to.append(pd.DataFrame({'Dist':aux,'Modelo':mod}))
    df_dist = pd.concat(dist_go_to)
    sns.kdeplot('Dist',data=df_dist,fill=True,ax=ax[1,1],hue='Modelo',palette=paleta)
    ax[1,1].set_xlabel('Agent\'s accumulated score')
    ax[1,1].set_title('Dist. of score')
    ax[1,1].get_legend().remove()
    # Plot efficiency
    efficiency = []
    for mod, grp in df.groupby('Modelo'):
        efficiency.append(pd.DataFrame({'Efficiency':P.encuentra_efficiency(grp),'Modelo':mod}))
    df_efficiency = pd.concat(efficiency)
    sns.kdeplot('Efficiency', data=df_efficiency,fill=True,ax=ax[1,2],hue='Modelo',palette=paleta)
    ax[1,2].set_xlabel('Efficiency')
    ax[1,2].set_title('Distribution of efficiency')
    ax[1,2].get_legend().remove()
    # Create the legend
    fig.legend(ax,     # The line objects
           labels=modelos[::-1],   # The labels for each line
           loc='lower center',   # Position of legend
           fancybox=True,
           shadow=True,
           ncol=2,
           borderaxespad=0.1,    # Small spacing around legend box
           title="Model"  # Title for the legend
           )

    plt.subplots_adjust(top=1.85)

def comparacion(df, degrees=0, comp='Model'):
    data = P.merge_modelos(df)
    try:
        a = data['Inaccuracy'].unique()
        variables = ['Attendance', 'Deviation', 'Efficiency', 'Gini', 'TBFI', 'Inaccuracy']
    except:
        variables = ['Attendance', 'Deviation', 'Efficiency', 'Gini', 'TBFI', 'Gap']
    fig, ax = plt.subplots(2,3, figsize=(9,6), tight_layout=True)
    for i, v in enumerate(variables):
        f = int(i/3)
        c = i % 3
        sns.lineplot(x='Modelo',y=v, data=data,ax=ax[f,c],err_style="bars",ci=95)
        ax[f,c].set_ylabel(f'Av. {v}')
        ax[f,c].set_xlabel(comp)
        ax[f,c].set_title(f'Av. {v} vs. ' + comp)
        ax[f,c].tick_params(labelrotation=degrees)

def graficar_influencias_modelos(df):
    data = P.merge_modelos(df)
    data = pd.DataFrame(data.groupby('Modelo').agg({'Attendance':'mean', 'Deviation':'mean', 'Efficiency':'mean', 'Gap':'mean', 'Gini':'mean'}).reset_index())
    variables = ['Efficiency', 'Gap', 'Gini']
    fig, ax = plt.subplots(2,3, figsize=(9,6), tight_layout=True)
    # Plot mean attendance vs variable
    for i, v in enumerate(variables):
        sns.regplot(x='Attendance',y=v,data=data,ax=ax[0,i],scatter_kws={'alpha':0.3})
        ax[0,i].set_ylabel(f'Av. {v}')
        ax[0,i].set_xlabel('Av. Attendance')
        ax[0,i].set_title(f'Mean attendance\nvs.\n{v}')
    # Plot std attendance vs variable
    for i, v in enumerate(variables):
        sns.regplot(x='Deviation',y=v,data=data,ax=ax[1,i],scatter_kws={'alpha':0.3})
        ax[1,i].set_ylabel(f'Av. {v}')
        ax[1,i].set_xlabel('Std.Dev Attendance')
        ax[1,i].set_title(f'Deviation attendance\nvs.\n{v}')

def graficar_heatmaps(df, parametros, variables, to_file=False):
    assert(len(parametros) == 2)
    p1 = parametros[0]
    p2 = parametros[1]
    assert(len(variables) < 7)
    data = P.merge_parametros(df, parametros, variables)
    fig, ax = plt.subplots(2,3, figsize=(9,6), tight_layout=True)
    for i, v in enumerate(variables):
        f = int(i/3)
        c = i % 3
        d = data.pivot_table(values=v,index=[p1],columns=[p2],aggfunc=np.mean)
        sns.heatmap(d,ax=ax[f,c])
        ax[f,c].set_title(v)
    if to_file:
        plt.savefig('heatmap.png')
        print("Imagen guardada!")


def dibuja_asistencia_vs(data, variable='Memoria'):
    Numero_agentes = max(data['Agente']) + 1
    aux = data.groupby([variable, 'Identificador', 'Ronda'])['Estado']\
        .sum().reset_index()
    aux.columns = [variable,
                   'Identificador',
                   'Ronda',
                   'Asistencia_total']
    aux['Asistencia_total'] = (aux['Asistencia_total']/Numero_agentes)*100
    rondas = aux['Ronda'].unique()
    aux1 = aux[aux['Ronda'] > rondas[-75]]
    aux1 = aux1.groupby([variable, 'Identificador'])['Asistencia_total']\
        .mean().reset_index()
    aux1.columns = [variable,
                   'Identificador',
                   'Asistencia_total']
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    for v, grp in aux.groupby(variable):
        sns.lineplot(x=grp['Ronda'], y=grp['Asistencia_total'], label=v, ax=ax[0])
    ax[0].legend().set_title(variable)
    sns.boxplot(x=aux1[variable], y=aux1['Asistencia_total'], ax=ax[1])
    ax[0].set_xlabel('Ronda')
    ax[0].set_ylabel('Asistencia promedio')
    ax[0].set_title('Asistencia promedio por ronda')
    ax[1].set_xlabel(variable)
    ax[1].set_ylabel('Asistencia últimas 75 rondas')
    ax[1].set_title('Distribución asistencia en las últimas 75 rondas')

def dibujar_puntaje_vs(data, variable):
    fig, ax = plt.subplots(2,1,figsize=(8,8))
    data_aux = data.groupby([variable, 'Identificador'])['Puntaje'].mean().reset_index()
    sns.boxplot(x=variable, y='Puntaje', data=data_aux, ax=ax[0])
    df = data.groupby([variable, 'Identificador', 'Agente'])['Puntaje'].mean().reset_index()
    for key, grp in df.groupby(variable):
        sns.distplot(grp['Puntaje'], ax=ax[1], label=key)
    ax[1].legend().set_title(variable)
    ax[0].set_ylabel('Puntaje')
    ax[0].set_title('Distribución puntaje vs ' + variable)
    ax[1].set_xlabel('Puntaje promedio')
    ax[1].set_ylabel('')
    ax[1].set_title('Distribución de la recompensa\n por cada ' + variable)
    fig.tight_layout()

def dibuja_usopredictores_vs(data, variable):
    df = data.groupby(variable)['Politica'].value_counts().rename_axis([variable, 'Politica']).reset_index(name='Usos')
    g = sns.FacetGrid(df, col=variable, aspect=1.5, height=4, sharex=False)
    g.map(sns.barplot, "Politica", "Usos")
    g.set_xticklabels(rotation=90)

def dibuja_puntajepredictor_vs(data, variable):
    data['Politica_lag'] = data.groupby([variable, 'Identificador', 'Agente'])['Politica'].transform('shift', -1)
    df = data.groupby([variable, 'Politica_lag', 'Identificador'])['Puntaje'].mean().reset_index()
    for p, Grp in df.groupby(variable):
        grp = Grp.sort_values(by='Puntaje')
        fig, ax = plt.subplots(1,1,figsize = (8,5))
        sns.swarmplot(x=grp['Politica_lag'], y=grp['Puntaje'])
        fig.suptitle(variable + ': ' + str(p), fontsize=14)
        plt.xticks(rotation=90)

def dibujar_violin(memoria, predictores, conectividad, espejos=True):
	columnas = ['Identificador','Ronda','Agente','Estado','Puntaje','Modelo']
	# Data ramdom
	data_random = leer_datos([0], [1], [0], verb=False, espejos=espejos)
#	data_random = data_random[data_random['Ronda']>79]
	data_random['Modelo'] = 'Random'
	data_random = data_random[columnas]
	# Data from model
	data_arthur = leer_datos([memoria], [predictores], [conectividad], verb=False, espejos=espejos)
#	data_arthur = data_arthur[data_arthur['Ronda']>79]
	data_arthur['Modelo'] = 'Arthur'
	data_arthur = data_arthur[columnas]
	# Concatentating
	data = pd.concat([data_random, data_arthur])
	# Helper dataframes
	Numero_agentes = max(data['Agente']) + 1
	aux = data.groupby(['Modelo','Identificador','Ronda'])['Estado']\
	        .sum().reset_index()
	aux.columns = ['Modelo','Identificador','Ronda','Asistencia_total']
	aux['Asistencia_total'] = (aux['Asistencia_total']/Numero_agentes)*100
	aux1 = data.groupby(['Modelo','Identificador','Agente'])['Puntaje'].sum().reset_index()
	aux1.columns = ['Modelo','Identificador','Agente','Puntaje_total']
	aux2 = data.groupby(['Modelo','Identificador','Agente'])['Estado'].sum().reset_index()
	aux2.columns = ['Modelo','Identificador','Agente','Estado_total']
	aux2['Estado_total'] = (aux2['Estado_total']/Numero_agentes)
	# Plotting
	fig, ax = plt.subplots(2,2,figsize=(8,8))
	sns.lineplot(data=aux,x='Ronda',y='Asistencia_total',hue='Modelo',ci=95,ax=ax[0,0])
	sns.violinplot(data=aux[aux.Ronda>79],x='Asistencia_total',y='Modelo',ax=ax[0,1])
	sns.violinplot(data=aux2,x='Estado_total',y='Modelo',ax=ax[1,0])
	sns.violinplot(data=aux1,x='Puntaje_total',y='Modelo',ax=ax[1,1])
	ax[0,0].set_xlabel('Round')
	ax[0,0].set_ylabel('Av. attendance')
	ax[0,0].set_ylim([40,105])
	ax[0,0].set_title('Average attendance \n over 100 simulations')
	ax[0,1].set_xlabel('Attendance (last 20 rounds)')
	ax[0,1].set_ylabel('Model')
	ax[0,1].set_title('Distribution of attendance per round \n over 100 simulations')
	ax[1,0].set_xlabel('Frequency of attendance')
	ax[1,0].set_title('Distribution of frequency of attendance \n per player and per simulation')
	ax[1,0].set_ylabel('Model')
	ax[1,1].set_xlabel('Accumulated score')
	ax[1,1].set_title('Distribution of accumulated score \n per player and per simulation')
	ax[1,1].set_ylabel('Model')
	fig.suptitle(f"Memoria={memoria}   Predictores={predictores}   Conectividad={conectividad}", fontsize=16)
	plt.tight_layout()

def dibuja_vs(data, variable):
    data1 = pd.DataFrame(data[data['Ronda']>25])
    Numero_agentes = max(data1['Agente']) + 1
    aux = data1.groupby([variable, 'Identificador', 'Ronda'])['Estado']\
        .sum().reset_index()
    aux.columns = [variable,
                   'Identificador',
                   'Ronda',
                   'Asistencia_total']
    aux['Asistencia_total'] = (aux['Asistencia_total']/Numero_agentes)*100
    rondas = aux['Ronda'].unique()
    aux1 = aux.groupby([variable, 'Identificador'])['Asistencia_total']\
        .mean().reset_index()
    aux1.columns = [variable,
                   'Identificador',
                   'Asistencia_total']
    data_aux = data1.groupby([variable, 'Identificador'])['Puntaje'].mean().reset_index()
    aux2 = aux.groupby([variable, 'Identificador'])['Asistencia_total']\
        .std().reset_index()
    aux2.columns = [variable,
                   'Identificador',
                   'Std_Asistencia_total']
    fig, ax = plt.subplots(3,1,figsize=(7,12))
    sns.boxplot(x=aux1[variable], y=aux1['Asistencia_total'], ax=ax[0], color='blue')
    sns.boxplot(x=variable, y='Std_Asistencia_total', data=aux2, ax=ax[1], color='cyan')
    sns.boxplot(x=variable, y='Puntaje', data=data_aux, ax=ax[2], color='red')
    if variable == 'Memoria':
        ax[2].set_xlabel('Memory')
    elif variable == 'Num_predic':
        ax[2].set_xlabel('Number of predictors')
    elif variable == 'Conectividad':
        ax[2].set_xlabel('Connectivity')
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    ax[0].set_ylabel('Attendance')
    ax[0].set_title('Distribution of attendance')
    ax[1].set_title('Distribution of Standard\n deviation of Attendance')
    ax[1].set_ylabel('Std. Attendance')
    ax[2].set_ylabel('Score')
    ax[2].set_title('Distribution of score')
    fig.tight_layout()

def analisis1(memoria, predictores, conectividad, espejos=True):
	data = leer_datos(memoria,predictores,conectividad,espejos=espejos,verb=False)
	grps_agente = data.groupby(['Identificador', 'Agente']).agg({'Estado':'mean', 'Puntaje':'mean'}).reset_index()
	sns.scatterplot(x='Estado',y='Puntaje',alpha=0.5,data=grps_agente)
	plt.suptitle('Agentes', fontsize=20)
	grps_politica = data.groupby(['Identificador', 'Politica']).agg({'Estado':'mean', 'Puntaje':'mean', 'Precision':'mean'}).reset_index()
	grps_politica['Espejo'] = grps_politica['Politica'].apply(lambda x: 'Yes' if 'espejo' in x else 'No')
	fig, ax = plt.subplots(1,2,figsize=(8,4))
	sns.scatterplot(x='Estado',y='Puntaje',hue='Espejo',alpha=0.25,data=grps_politica,ax=ax[0])
	sns.scatterplot(x='Precision',y='Puntaje',hue='Espejo',alpha=0.25,data=grps_politica,ax=ax[1])
	plt.suptitle('Políticas', fontsize=20)
	plt.tight_layout()

def analizar_politicas(memoria, predictores, conectividad, criterio, espejos=True):
	data = leer_datos(memoria,predictores,conectividad,espejos=espejos,verb=False)
	grps_politica = data.groupby(['Identificador', 'Politica']).agg({'Estado':'mean', 'Puntaje':'mean', 'Precision':'mean'}).reset_index()
	grps_politica = grps_politica.query(criterio)
	grps_politica['Espejo'] = grps_politica['Politica'].apply(lambda x: 'Yes' if 'espejo' in x else 'No')
	grps_politica['Ventana'] = grps_politica['Politica'].apply(lambda x: int(x.split('-')[0]))
	grps_politica['Tipo'] = grps_politica['Politica'].apply(lambda x: x.split('-')[1].split('(')[0])
	fig, ax = plt.subplots(1,3,figsize=(12,4))
	ventanas = grps_politica.Ventana.value_counts().reset_index()
	sns.barplot(x='index', y='Ventana',data=ventanas,ax=ax[0])
	tipos = grps_politica.Tipo.value_counts().reset_index()
	sns.barplot(x='index', y='Tipo',data=tipos,ax=ax[1])
	espejos = grps_politica.Espejo.value_counts().reset_index()
	sns.barplot(x='index', y='Espejo',data=espejos,ax=ax[2])
	ax[0].set_xlabel('')
	ax[1].set_xlabel('')
	ax[2].set_xlabel('')
	ax[0].set_ylabel('Frequency')
	ax[1].set_ylabel('')
	ax[2].set_ylabel('')
	ax[0].set_title('Ventanas')
	ax[1].set_title('Tipos')
	ax[2].set_title('Espejos')
	plt.tight_layout()
