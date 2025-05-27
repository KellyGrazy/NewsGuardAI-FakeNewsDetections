import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

# Carregar os arquivos CSV com as notícias falsas e verdadeiras
noticias_falsas = pd.read_csv('fake.csv')
noticias_verdadeiras = pd.read_csv('true.csv')

# Adicionar uma coluna para identificar se a notícia é falsa ou verdadeira
noticias_falsas['rotulo'] = 0  # 0 significa Fake News
noticias_verdadeiras['rotulo'] = 1  # 1 significa Notícia verdadeira

# Juntar os dois datasets em um único
noticias = pd.concat([noticias_falsas, noticias_verdadeiras], ignore_index=True)

# Criar um dicionário para converter os meses para número
meses = {mes: indice for indice, mes in enumerate([
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}

# Função para converter datas do formato 'July 31, 2017' para datetime
def formatar_data(data_texto):
    try:
        partes = data_texto.replace(',', '').split()
        return datetime(int(partes[2]), meses[partes[0]], int(partes[1]))
    except:
        return None  # Retorna None caso a data esteja errada

# Aplicar a função nas datas
noticias['data'] = noticias['date'].apply(formatar_data)

# Remover valores nulos e duplicatas
noticias.dropna(inplace=True)
noticias.drop_duplicates(subset=['title', 'text'], inplace=True)

# Lista de stopwords básicas
stopwords_personalizadas = set(["a", "an", "and", "the", "to", "in", "for", "on", "at", "with", "by", "of", "is", "it", "that", "this", "as", "are", "was", "were", "be", "been", "being", "has", "have", "had", "having", "do", "does", "did", "doing", "but", "or", "if", "because", "so", "out", "up", "down", "from", "about", "over", "under", "then", "there", "after", "before", "while", "during", "more", "most", "some", "such", "no", "nor", "not", "only", "own", "same", "can", "will", "just", "should", "now"])

# Função para pré-processamento de texto
def limpar_texto(texto):
    texto = texto.lower()  # Converter para minúsculas
    texto = re.sub(r'[^a-zA-Z]', ' ', texto)  # Remover caracteres especiais
    palavras = texto.split()  # Tokenizar palavras manualmente
    palavras = [palavra for palavra in palavras if palavra not in stopwords_personalizadas]  # Remover stopwords
    return ' '.join(palavras)

# Aplicar pré-processamento ao texto
noticias['texto_limpo'] = (noticias['title'] + ' ' + noticias['text']).apply(limpar_texto)

# Análise exploratória para entender o formato dos dados
print(noticias.info())  # Exibir informações gerais do dataset
print(noticias['rotulo'].value_counts())  # Contar quantas notícias falsas e verdadeiras existem

# Visualizar graficamente a distribuição das notícias
sns.countplot(x=noticias['rotulo'])
plt.title('Quantidade de Notícias Falsas e Verdadeiras')
plt.show()

# Analisando os subjects mais comuns em notícias falsas e verdadeiras
subjects_falsas = noticias_falsas['subject'].value_counts().head(10)
subjects_verdadeiras = noticias_verdadeiras['subject'].value_counts().head(10)

print("Top 10 assuntos mais comuns em Fake News:")
print(subjects_falsas)
print("\nTop 10 assuntos mais comuns em Notícias Verdadeiras:")
print(subjects_verdadeiras)

# Plotando gráficos para visualizar
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.barplot(x=subjects_falsas.values, y=subjects_falsas.index)
plt.title("Principais Assuntos - Fake News")

plt.subplot(1, 2, 2)
sns.barplot(x=subjects_verdadeiras.values, y=subjects_verdadeiras.index)
plt.title("Principais Assuntos - Notícias Verdadeiras")

plt.tight_layout()
plt.show()

# Preparação do texto para alimentar o modelo de machine learning
# Vamos transformar os textos em números usando o TfidfVectorizer
vetorizador = TfidfVectorizer(max_features=5000)
X = vetorizador.fit_transform(noticias['texto_limpo'])  # Transformar texto pré-processado em vetores

y = noticias['rotulo']  # Define os rótulos como variável alvo

# Separação dos dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão logística
modelo = LogisticRegression()
modelo.fit(X_treino, y_treino)

# Fazer previsões no conjunto de teste
y_predito = modelo.predict(X_teste)

# Calcular a acurácia do modelo
acuracia = accuracy_score(y_teste, y_predito)
print(f'Acurácia do Modelo: {acuracia:.4f}')

# Mostrar métricas detalhadas do desempenho do modelo
print("Relatório de Classificação:")
print(classification_report(y_teste, y_predito))

# Matriz de confusão para visualizar os erros
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_teste, y_predito), annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

import streamlit as st
import joblib

# Carregar modelo e vetor
model = joblib.load("news_guard_model.joblib")
vectorizer = joblib.load("news_guard_vectorizer.joblib")

st.set_page_config(page_title="NewsGuard AI", layout="centered")
st.title("🧠 NewsGuard AI")
st.write("Classificação de notícias como **falsas** ou **reais** usando inteligência artificial.")

user_input = st.text_area("Cole aqui o conteúdo da notícia:", height=200)

if st.button("Verificar"):
    if user_input.strip() == "":
        st.warning("Digite algum conteúdo para analisar.")
    else:
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]
        proba = model.predict_proba(text_vec).max()

        if prediction == "real":
            st.success(f"✅ Esta notícia parece **VERDADEIRA** com confiança de {proba:.2%}.")
        else:
            st.error(f"⚠️ Esta notícia parece **FALSA** com confiança de {proba:.2%}.")
streamlit run app.py
import streamlit as st
import joblib

# Carregar modelo e vetor
model = joblib.load("news_guard_model.joblib")
vectorizer = joblib.load("news_guard_vectorizer.joblib")

st.set_page_config(page_title="NewsGuard AI", layout="centered")
st.title("🧠 NewsGuard AI")
st.write("Classificação de notícias como **falsas** ou **reais** usando inteligência artificial.")

user_input = st.text_area("Cole aqui o conteúdo da notícia:", height=200)

if st.button("Verificar"):
    if user_input.strip() == "":
        st.warning("Digite algum conteúdo para analisar.")
    else:
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]
        proba = model.predict_proba(text_vec).max()

        if prediction == "real":
            st.success(f"✅ Esta notícia parece **VERDADEIRA** com confiança de {proba:.2%}.")
        else:
            st.error(f"⚠️ Esta notícia parece **FALSA** com confiança de {proba:.2%}.")


