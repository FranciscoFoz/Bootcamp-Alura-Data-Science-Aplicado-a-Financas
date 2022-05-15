import streamlit as st
from joblib import load
import pandas as pd
from utils import Transformador

#Cor de fundo do listbox
st.markdown('<style>div[role="listbox"] ul{background-color: #eee1f79e};</style>', unsafe_allow_html=True)


def avaliar_mau(dict_respostas):    
    modelo = load('Objetos/Credit_scoring_RandomForest_preditor.joblib')
    features = load('Objetos/features.joblib')

    if dict_respostas['Anos_desempregado'] > 0:
        dict_respostas['Anos_empregado'] = dict_respostas['Anos_desempregado'] * -1 
	
    respostas = []
    for coluna in features:
        respostas.append(dict_respostas[coluna])

    df_novo_cliente = pd.DataFrame(data=[respostas], columns=features)

    mau = modelo.predict(df_novo_cliente)[0]

    return mau

	



st.image('https://github.com/FranciscoFoz/Bootcamp-Alura-Data-Science-Aplicado-a-Financas/raw/main/Imagens/bytebank_logo.png')
st.write('# Simulador de Avaliação de crédito')

expander_trabalho = st.beta_expander("Trabalho")

expander_pessoal = st.beta_expander("Pessoal")

expander_familia = st.beta_expander("Familia")

dict_respostas = {}
lista_campos = load('Objetos/lista_campos.joblib')

with expander_trabalho:

	col1_form, col2_form = st.beta_columns(2)

	dict_respostas['Categoria_de_renda'] = col1_form.selectbox('Qual a sua categoria de renda ?', lista_campos['Categoria_de_renda'])

	dict_respostas['Ocupacao'] = col1_form.selectbox('Qual a sua ocupação ?', lista_campos['Ocupacao'])

	dict_respostas['Tem_telefone_trabalho'] = 1 if col1_form.selectbox('Você tem um telefone de trabalho ?', ['Sim', 'Não']) == 'Sim' else 0

	dict_respostas['Rendimento_Anual'] = col2_form.slider('Qual o seu salário mensal ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=35000, step=500) * 12

	dict_respostas['Anos_empregado'] = col2_form.slider('Quantos anos você está empregado(a) ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=50, step=1)

	dict_respostas['Anos_desempregado'] = col2_form.slider('Quantos anos você está desempregado(a) ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=50, step=1)

with expander_pessoal:

    col3_form, col4_form = st.beta_columns(2)

    dict_respostas['Idade'] = col3_form.slider('Qual a sua idade ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=100, step=1)

    dict_respostas['Grau_Escolaridade'] = col3_form.selectbox('Qual o seu grau de escolaridade ?', lista_campos['Grau_Escolaridade'])

    dict_respostas['Estado_Civil'] = col3_form.selectbox('Qual o seu estado civil ?', lista_campos['Estado_Civil'])

    dict_respostas['Tem_Carro'] = 1 if col4_form.selectbox('Você tem um carro ?', ['Sim', 'Não']) == 'Sim' else 0

    dict_respostas['Tem_telefone_fixo'] = 1 if col4_form.selectbox('Você tem um telefone fixo ?', ['Sim', 'Não']) == 'Sim' else 0

    dict_respostas['Tem_email'] = 1 if col4_form.selectbox('Você tem um email ?', ['Sim', 'Não']) == 'Sim' else 0


with expander_familia:

    col4_form, col5_form = st.beta_columns(2)

    dict_respostas['Moradia'] = col4_form.selectbox('Qual o seu tipo de moradia ?', lista_campos['Moradia'])

    dict_respostas['Tem_Casa_Propria'] = 1 if col4_form.selectbox('Você tem casa própria ?', ['Sim', 'Não']) == 'Sim' else 0

    dict_respostas['Tamanho_Familia'] = col5_form.slider('Qual o tamanho da sua família ?', help='Podemos mover a barra usando as setas do teclado', min_value=1, max_value=20, step=1)

    dict_respostas['Qtd_Filhos'] = col5_form.slider('Quantos filhos você tem ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=20, step=1)

if st.button('Avaliar crédito'):
	if avaliar_mau(dict_respostas):
		st.error('Crédito negado')
	else:
		st.success('Crédito aprovado')