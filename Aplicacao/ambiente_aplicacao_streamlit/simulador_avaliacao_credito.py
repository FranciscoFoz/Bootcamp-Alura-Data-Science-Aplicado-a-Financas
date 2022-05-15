import streamlit as st
from joblib import load
import pandas as pd
from utils import Transformador


def avaliar_mau(dict_respostas):
    modelo = load('https://github.com/FranciscoFoz/Bootcamp-Alura-Data-Science-Aplicado-a-Financas/raw/main/Objetos/Credit_scoring_RandomForest_preditor.joblib')
    features = load('https://github.com/FranciscoFoz/Bootcamp-Alura-Data-Science-Aplicado-a-Financas/raw/main/Objetos/features.joblib')

    if dict_respostas['Anos_desempregado'] > 0:
        dict_respostas['Anos_empregado'] = dict_respostas['Anos_desempregado'] * -1

    respostas = []
    for coluna in features:
        respostas.append(dict_respostas[coluna])

    df_novo_cliente = pd.DataFrame(data=[respostas],columns=features)

    mau = modelo.predict(df_novo_cliente)[0]

    return modelo


st.image('https://raw.githubusercontent.com/FranciscoFoz/Bootcamp-Alura-Data-Science-Aplicado-a-Financas/main/Imagens/bytebank_logo.png')
st.write('# Simulador de Avaliação de crédito')

my_expander_trabalho = st.beta_expander('Trabalho')

my_expander_pessoal = st.beta_expander('Pessoal')

my_expander_familia = st.beta_expander('Familia')


dict_respostas = {}
lista_campos = load('https://github.com/FranciscoFoz/Bootcamp-Alura-Data-Science-Aplicado-a-Financas/raw/main/Objetos/lista_campos.joblib')

with my_expander_trabalho:
    col1_form, col2_form = st.beta_columns(2)

    dict_respostas['Categoria_de_renda'] = col1_form.selectbox('Qual é a sua categoria de renda ?', lista_campos['Categoria_de_renda'])

    dict_respostas['Ocupacao'] = col1_form.selectbox('Qual é a sua ocupação ?', lista_campos['Ocupacao'])

    dict_respostas['Tem_telefone_trabalho'] = 1 if col1_form.selectbox('Você tem um telefone de trabalho ?', ['Sim', 'Não']) == 'Sim' else 0

    dict_respostas['Rendimento_Anual'] = col2_form.slider('Qual o seu salário salário mensal ?',help='Podemos mover a barra usando as setas do teclado', min_value = 0, max_value =35000, step = 500) * 12

    dict_respostas['Anos_empregado'] = col2_form.slider('Quantos anos você está empregado ?',help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=50, step=1)

    dict_respostas['Anos_desempregado'] = col2_form.slider('Quantos anos você está desempregado ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=50, step=1)

    


with my_expander_pessoal:

    col3_form, col4_form = st.beta_columns(2)

    dict_respostas['Idade'] = col3_form.slider('Qual é a sua idade ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=100, step=1)

    dict_respostas['Grau_Escolaridade'] = col3_form.selectbox('Qual é o seu grau de Escolaridade ?', lista_campos['Grau_Escolaridade'])

    dict_respostas['Estado_Civil'] = col3_form.selectbox('Qual o seu estado civil ?', lista_campos['Estado_Civil'])

    dict_respostas['Tem_Carro'] = 1 if col4_form.selectbox('Você tem um carro ?', ['Sim', 'Não']) == 'Sim' else 0

    dict_respostas['Tem_telefone_fixo'] = 1 if col4_form.selectbox('Você tem um telefone fixo ?', ['Sim', 'Não']) == 'Sim' else 0

    dict_respostas['Tem_email'] = 1 if col4_form.selectbox('Você tem um email ?', ['Sim', 'Não']) == 'Sim' else 0


with my_expander_familia:

    col4_form, col5_form = st.beta_columns(2)

    dict_respostas['Moradia'] = col4_form.selectbox('Qual é o seu tipo de moradia ?', lista_campos['Moradia'])

    dict_respostas['Tem_Casa_Propria'] = 1 if col4_form.selectbox('Você tem casa propria ?', ['Sim', 'Não']) == 'Sim' else 0

    dict_respostas['Tamanho_Familia'] = col5_form.slider('Qual o tamanho da sua família ?', help='Podemos mover a barra usando as setas do teclado', min_value=1, max_value=20, step=1)

    dict_respostas['Qtd_Filhos'] = col5_form.slider('Quantos filhos você tem? ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=20, step=1)

if st.button('Avaliar crédito'):
    if avaliar_mau(dict_respostas):
        st.error('Crédito negado.')
    else:
        st.sucess('Crédito aprovado!')
