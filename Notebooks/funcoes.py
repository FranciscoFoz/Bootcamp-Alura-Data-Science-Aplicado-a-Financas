"""
Projeto Bootcamp-Alura-Data-Science-Aplicado-a-Financas

Funcoes do projeto
"""
#Importações

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, plot_confusion_matrix, roc_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from scipy import stats

import six
import sys
sys.modules['sklearn.externals.six'] = six

import lazypredict
from lazypredict.Supervised import LazyClassifier, LazyRegressor

from skopt import BayesSearchCV

from joblib import dump



#Funcoes do primeiro notebook

#Funcoes de transformações de dados
def verifica(registros):
    """
    Esta função irá verificar cada cliente pela faixa de atraso e atribuir 1 se
    ele tiver atraso com mais de 60 dias e 0 se não tiver.
    """
    lista_status = registros['Faixa_atraso'].to_list()
    if '60-89 dias' in lista_status or '90-119 dias' in lista_status or '120-149 dias' in lista_status or '>150 dias' in lista_status:
        return 1
    else:
        return 0

def transforma_outliers(dados):
  """
  Esta função tem como objetivo substituir os valores de outliers das colunas: 
  'Qtd_Filhos','Rendimento_Anual','Tamanho_Familia','Anos_empregado', pelos 
  valores de limite de 90% de cada distribuição.
  """
  
  lista_variaveis = ['Qtd_Filhos','Rendimento_Anual','Tamanho_Familia','Anos_empregado']        #Listar variáveis

  for variavel in lista_variaveis:                                                              #For para cada variável dentro da lista
    limite_superior = dados[variavel].quantile(0.9)                                             #Definir o valor do limite superior com 90% do valor da distribuição.
    dados[variavel] = dados[variavel].mask(dados[variavel] > limite_superior,limite_superior)   #Subsititur todos os valores acima do limite superior, pelo limite superior

  return dados    

#Funcoes de visualização de dados

def boxplot_bons_maus(dados,variavel,titulo):

  '''
  Esta função irá plotar um gráfico de boxplot para uma variável escolhida 
  dentro do DataFrame preparado.

  dados = conjunto de dados transformado para o modelo
  variavel = variável do conjunto de dados a se visualizar
  titulo = título do gráfico
  '''

  sns.set_context('notebook', font_scale=1.1, rc={'lines.linewidth': 1})        #Configurar o tema do gráfico
  sns.set_style("darkgrid")                                                     #Configurar o fundo do gráfico


  fig = plt.figure(figsize=(10,8))                                              #Configura o tamnho do gráfico
  ax = sns.boxplot(y=variavel,                                                  #Plota o boxplot com a variável escolhida
                   x=dados.Mau.map({0:'Bons',1:'Maus'}),                        #Mapeia os valores da coluna Mau, classificando-os como Bons e Maus
                   data=dados,                                                  #Configura a fonte de dados do gráfico
                   palette='blend:#457EB3,#B33328')                             #Escolhe as cores
  
  if variavel == 'Rendimento_Anual':
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('R${x:,.0f}'))       #Formatar números do eixo y
 
        
  plt.xlabel(None)                                                              #Exclui o rótulo do eixo x
  plt.ylabel(None)                                                              #Exclui o rótulo do eixo y0
  plt.title(titulo + '\n',size=20,loc='left')                                   #Configura o título do gráfico
  plt.show()                                                                    #Mostra o gráfico


def grafico_barras_vertical_bons_maus(bons_clientes,maus_clientes,variavel,titulo,condicao_variavel_positiva,condicao_variavel_negativa):

  '''
  Esta função irá plotar um gráfico de barras para uma variável escolhida 
  dentro dos DataFrames preparados.

  bons_clientes = dados filtrados apenas com clientes Mau ==0
  maus_clientes = dados filtrados apenas com clientes Mau ==1
  variavel = nome da variavel do Dataframe
  titulo = titulo do gráfico
  condicao_variavel_positiva = nome da condição da variável positiva (exemplo: Tem carro)
  condicao_variavel_negativa = nome da condição da variável negativa (exemplo: Não tem carro)
  '''
 
  df_bons = bons_clientes[[variavel]]                                                                    #Seleciona apenas os valores da coluna
  df_bons = round((df_bons.value_counts(normalize=True)*100),2).reset_index()                            #Conta os valores em % arredondado em 2 casas     
  df_bons[variavel] = df_bons[variavel].map({1:condicao_variavel_positiva,0:condicao_variavel_negativa}) #Mapeia os valores conforme as condições passadas
  df_bons.columns = [variavel,'Quantidade']                                                              #Renomeia as colunas 

  df_maus = maus_clientes[[variavel]]                                                                    #Seleciona apenas os valores da coluna
  df_maus = round((df_maus.value_counts(normalize=True)*100),2).reset_index()                            #Conta os valores em % arredondado em 2 casas     
  df_maus[variavel] = df_maus[variavel].map({1:condicao_variavel_positiva,0:condicao_variavel_negativa}) #Mapeia os valores conforme as condições passadas
  df_maus.columns = [variavel,'Quantidade']                                                              #Renomeia as colunas 



  sns.set_context('notebook', font_scale=1.3, rc={'lines.linewidth': 1})                                 #Configura o tema do gráfico
  sns.set_style("white")                                                                                 #Configurar o fundo do gráfico



  fig = plt.figure(figsize=(12,6))                                                                       #Configura o tamanho da figura

  plt.subplot(1, 2, 1)                                                                                   #Condigura o 1º gráfico na coluna 1
  ax1 = sns.barplot(data=df_bons,y='Quantidade',x=variavel,                                              #Plota o gráfico
                    palette='blend:#457EB3,#284866')                                                     #Escolhe a cor
  ax1.set_xlabel('Bons clientes')                                                                        #Configura o rótulo do eixo x do primeiro gráfico
  for spine in plt.gca().spines.values():                                                                #Remover os eixos do gráfico
    spine.set_visible(False)
    
  plt.bar_label(ax1.containers[0],fmt='%.2f%%')                                                          #Adiciona rótulos nas barras

  plt.title(titulo + '\n',size=20,loc='left')                                                            #Configura o título do gráfico


  plt.subplot(1, 2, 2)                                                                                   #Condigura o 1º gráfico na coluna 1
  ax2 = sns.barplot(data=df_maus,y='Quantidade',x=variavel,                                              #Plota o gráfico
                    palette='blend:#B33328,#591813')                                                     #Escolhe a cor
  ax2.set_xlabel('Maus clientes')                                                                        #Configura o rótulo do eixo x do segundo gráfico
  for spine in plt.gca().spines.values():                                                                #Remover os eixos do gráfico
    spine.set_visible(False)

  plt.bar_label(ax2.containers[0],fmt='%.2f%%')                                                          #Adiciona rótulos nas barras



  for ax in ax1,ax2:                                                                                     #For para os dois gráficos 
    ax.set_ylabel(None)                                                                                  #Exclui o rótulo do eixo y
    ax.set_yticks([])                                                                                    #Exclui valores do eixo y
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x}%'))                                      #Formatar números do eixo y

  
  plt.show()                                                                                             #Mostra o gráfico

def grafico_barras_horizontal_bons_maus(bons_clientes,maus_clientes,variavel,titulo):

  '''
  Esta função irá plotar um gráfico de barras para uma variável escolhida 
  dentro dos DataFrames preparados.

  bons_clientes = dados filtrados apenas com clientes Mau == 0
  maus_clientes = dados filtrados apenas com clientes Mau == 1
  variavel = nome da variavel do Dataframe
  titulo = titulo do gráfico
  condicao_variavel_positiva = nome da condição da variável positiva (exemplo: Tem carro)
  condicao_variavel_negativa = nome da condição da variável negativa (exemplo: Não tem carro)
  '''
 
  df_bons = bons_clientes[[variavel]]                                                                    #Seleciona apenas os valores da coluna
  df_bons = round((df_bons.value_counts(normalize=True)*100),2).reset_index()                            #Conta os valores em % arredondado em 2 casas     
  df_bons.columns = [variavel,'Quantidade']                                                              #Renomeia as colunas 

  df_maus = maus_clientes[[variavel]]                                                                    #Seleciona apenas os valores da coluna
  df_maus = round((df_maus.value_counts(normalize=True)*100),2).reset_index()                            #Conta os valores em % arredondado em 2 casas    
  df_maus.columns = [variavel,'Quantidade']                                                              #Renomeia as colunas 



  sns.set_context('notebook', font_scale=1.3, rc={'lines.linewidth': 1})                                 #Configura o tema do gráfico
  sns.set_style("white")                                                                                 #Configurar o fundo do gráfico



  fig = plt.figure(figsize=(15,6))                                                                       #Configura o tamanho da figura

  plt.subplot(1, 3, 1)                                                                                   #Condigura o 1º gráfico na coluna 1
  ax1 = sns.barplot(data=df_bons,y=variavel,x='Quantidade',                                              #Plota o gráfico
                    palette='blend:#457EB3,#284866',
                    order=bons_clientes[variavel].value_counts().index)                                  #Escolhe a cor
  ax1.set_xlabel('Bons clientes')                                                                        #Configura o rótulo do eixo x do primeiro gráfico
  for spine in plt.gca().spines.values():                                                                #Remover os eixos do gráfico
    spine.set_visible(False)
    
  plt.bar_label(ax1.containers[0],fmt='%.2f%%',padding=5)                                                #Adiciona rótulos nas barras

  plt.title(titulo + '\n',size=20,loc='left')                                                            #Configura o título do gráfico


  plt.subplot(1, 3, 3)                                                                                   #Condigura o 1º gráfico na coluna 1
  ax2 = sns.barplot(data=df_maus,y=variavel,x='Quantidade',                                              #Plota o gráfico
                    palette='blend:#B33328,#591813',
                    order=maus_clientes[variavel].value_counts().index)                                  #Escolhe a cor
  ax2.set_xlabel('Maus clientes')                                                                        #Configura o rótulo do eixo x do segundo gráfico
  for spine in plt.gca().spines.values():                                                                #Remover os eixos do gráfico
    spine.set_visible(False)

  plt.bar_label(ax2.containers[0],fmt='%.2f%%',padding=5)                                                #Adiciona rótulos nas barras



  for ax in ax1,ax2:                                                                                     #For para os dois gráficos 
    ax.set_ylabel(None)                                                                                  #Exclui o rótulo do eixo y
    ax.set_xticks([])                                                                                    #Exclui valores do eixo y
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x}%'))                                      #Formatar números do eixo y

  
  plt.show()                                                                                             #Mostra o gráfico


#Funcoes do segundo notebook

def escolhe_modelo(dados,SEED):

    """
    Esta função irá retornar as métricas de avalição de cada modelo, para que  
    seja escolhido dentre os melhores.

    dados: dados que serão transformados dentro da função.

    SEED: valor para manter a reprodutibilidade do modelo
    """

    # separando dados em x e y, e tambem removendo a coluna de ID_cliente
    x = dados.drop(['ID_Cliente', 'Mau'], axis=1)
    y = dados.drop('ID_Cliente', axis=1)['Mau']

    # Separando dados em treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=SEED)

    # Classificando 
    classifiers = LazyClassifier(ignore_warnings=True, custom_metric=None)

    models,predictions = classifiers.fit(x_train, x_test, y_train, y_test)
    

    return models

def roda_modelo(modelo, dados,SEED):

    """
    Esta função rodará o modelo, imprimirá o AUC médio e 
    o relatório da classificação.

    modelo: modelo que será rodado dentro da função.

    dados: dados que serão transformados dentro da função.

    SEED: valor para manter a reprodutibilidade do modelo
    """


    # separando dados em x e y, e tambem removendo a coluna de ID_cliente
    x = dados.drop(['ID_Cliente', 'Mau'], axis=1)
    y = dados.drop('ID_Cliente', axis=1)['Mau']
    
    # Separando dados em treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=SEED)
    # Treinando modelo com os dados de treino
    modelo.fit(x_train, y_train)
    # Calculando a probabilidade e calculando o AUC
    prob_predic = modelo.predict_proba(x_test)
    auc = roc_auc_score(y_test, prob_predic[:,1])
    print(f"AUC {auc}")
    # Curva ROC
    tfp, tvp, limite = roc_curve(y_test, prob_predic[:,1])
    plt.subplots(1,figsize=(6,6))
    plt.title('Curva ROC')
    plt.plot(tfp,tvp)
    plt.plot([0, 1], ls="--", c = 'red')                                                   # Plotando linha para uma curva ROC a 
    plt.plot([0, 0], [1, 0], ls="--", c = 'green'), plt.plot([1, 1], ls="--", c = 'green') # Plotando linha para a curva ROC perfeita
    plt.ylabel('Sensibilidade')
    plt.xlabel('Especificidade')
    plt.show()
    # Separando a probabilidade de ser bom e mau, e calculando o KS
    print(100*'-')
    data_bom = np.sort(modelo.predict_proba(x_test)[:, 1])
    data_mau = np.sort(modelo.predict_proba(x_test)[:, 0])
    kstest = stats.ks_2samp(data_bom, data_mau)
    print(f"KS {kstest}")
    # Criando matriz de confusão
    print(100*'-')
    fig, ax = plt.subplots(figsize=(6,6))
    matriz_confusao = plot_confusion_matrix(modelo, x_test, y_test, values_format='.0f', display_labels=['Bons', 'Maus'], cmap = 'Blues',ax=ax)
    plt.ylabel('Valores reais')
    plt.xlabel('Valores preditos')
    plt.grid(False)
    plt.title('Matriz de confusão')    
    plt.show(matriz_confusao)
    # Fazendo a predicao dos dados de teste e calculando o classification report
    print(100*'-')
    predicao = modelo.predict(x_test)
    print("\nClassification Report")
    print(classification_report(y_test, predicao, zero_division=0))


def metricas_do_modelo(modelo,nome_modelo,dados,SEED):

  '''
  Esta função rodará o modelo e retornará as principais métricas dele.

  modelo: modelo que será rodado dentro da função.

  nome_modelo: nome do modelo que sairá nas métricas.

  dados: dados que serão transformados dentro da função.

  SEED: valor para manter a reprodutibilidade do modelo
  '''

  # separando dados em x e y, e tambem removendo a coluna de ID_cliente
  x = dados.drop(['ID_Cliente', 'Mau'], axis=1)
  y = dados.drop('ID_Cliente', axis=1)['Mau']
    
  # Separando dados em treino e teste
  x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=SEED)

  # Treinando modelo com os dados de treino
  modelo.fit(x_train, y_train)

  # Calculando a probabilidade e calculando o AUC
  prob_predic = modelo.predict_proba(x_test)
  auc = roc_auc_score(y_test, prob_predic[:,1])

  # Fazendo a predicao dos dados e obtendo as métricas
  predicao = modelo.predict(x_test)
  metricas = classification_report(y_test, predicao, zero_division=0,output_dict=True)
  metricas = pd.DataFrame(metricas)

  df_metricas = pd.DataFrame({'AUC': auc,
                              'F1-score_0': [metricas.iloc[2,0]],
                              'F1-score_1': [metricas.iloc[2,1]],
                              'Acurácia' : [metricas.iloc[0,2]]},
                               index=[nome_modelo])

  return df_metricas


class Transformador(BaseEstimator, TransformerMixin):
    def __init__(self, colunas_quantitativas, colunas_categoricas):
        self.colunas_quantitativas = colunas_quantitativas
        self.colunas_categoricas = colunas_categoricas
        self.enc = OneHotEncoder()
        self.scaler = MinMaxScaler()

    def fit(self, X, y = None ):
        self.enc.fit(X[self.colunas_categoricas])
        self.scaler.fit(X[self.colunas_quantitativas])
        return self 

    def transform(self, X, y = None):
      
      X_categoricas = pd.DataFrame(data=self.enc.transform(X[self.colunas_categoricas]).toarray(),
                                  columns= self.enc.get_feature_names(self.colunas_categoricas))
      
      X_quantitativas = pd.DataFrame(data=self.scaler.transform(X[self.colunas_quantitativas]),
                                  columns= self.colunas_quantitativas)
      
      X = pd.concat([X_quantitativas, X_categoricas], axis=1)

      return X

def tabela_comparativa(lista_dos_DataFrames):
  '''

  lista_dos_DataFrames = lista com todos os DataFrames a serem concatenados.

  Esta função tem como objetivo gerar um DataFrame com as métricas dos modelos.
  Ordenando os valores pelo AUC e pelo f1-Score dos valores 1.
  '''

  tabela_comparativa_das_metricas = pd.concat(lista_dos_DataFrames)
  tabela_comparativa_das_metricas = tabela_comparativa_das_metricas.sort_values(by=['AUC','F1-score_1'],ascending=False)


  return tabela_comparativa_das_metricas


def otimizar_param_bayesiano(modelo,params,dados):

    '''
    Esta função irá gerar os melhores hiperparamentros 
    através de um cálculo da estimativa bayesiana.
    
    modelo: modelo que será rodado dentro da função.
    
    paramns: parametros que serão testados no modelo para verificar qual é o melhor.

    dados: dados que serão transformados dentro da função.
    '''
    
    # separando dados em x e y, e tambem removendo a coluna de ID_cliente
    x = dados.drop(['ID_Cliente', 'Mau'], axis=1)
    y = dados.drop('ID_Cliente', axis=1)['Mau']
    
    # Separando dados em treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=634413)
    
    #Otimizando o modelo
    opt = BayesSearchCV(
        modelo,
        params)

    opt.fit(x_train, y_train)

    return opt.best_params_

