PROJETO APLICADO II


Projeto de Ciência de Dados - Detecção de Fake News em Notícias Online
PROFESSOR: FELIPE ALBINO DOS SANTOS

GRUPO:
ANTONIO H CALDAS MELLO – 10433799– 10433799@mackenzista.com.br
KELLY GRAZIELY PENA – 110416108 – 10416108@mackenzista.com.br









São Paulo
2025
SUMÁRIO

1.	Premissas do projeto	3
2.	Objetivos e metas	3
3.	Cronograma de atividades	4
4.	Bibliotecas e repositório no github	4
5.	Repositório no github
6.	Base de Dados e Análise Exploratória	5
7.	Tratamento da Base de Dados	6
8.	Calculo da Acurácia	6
9.	Resultados Preliminares
10.	Produto gerado e Modelo de Negócios	7
11.	Esboço do storytelling
12.	Proxima etapa
13.	Referência bibliográfica




 
1.	PREMISSAS DO PROJETO
Nome da Empresa: NewsGuard AI

Área de Atuação: Jornalismo e Tecnologia

Descrição: A NewsGuard AI é uma startup focada em detectar fake news usando inteligência artificial, ajudando leitores e veículos de comunicação a identificarem informações falsas.

Fonte de Dados: Bases públicas de notícias verdadeiras e falsas, como o Fake News Corpus ou Liar Dataset.

Tipo de Dados: Texto (notícias e manchetes publicadas online).

2.	OBJETIVOS E METAS
Objetivo Geral: Desenvolver um modelo de aprendizado de máquina para classificar notícias como verdadeiras ou falsas.
Metas Específicas:
Coletar e preparar uma base de notícias reais e fake news.
Implementar técnicas de processamento de linguagem natural (NLP) para analisar os textos.
Treinar modelos de aprendizado de máquina para detectar padrões em fake news.
Criar um painel interativo para visualizar a confiabilidade das notícias.





3. CRONOGRAMA

|Etapa 1  |	|Definição do escopo do projeto                 |1 semana   |
|Etapa 2  |	|Coleta e limpeza dos dados	                    |2 semanas  |
|Etapa 3  |	Exploração e visualização dos dados             |1 semana   |
|Etapa 4  |	Implementação dos modelos de classificação      |3 semanas  |
|Etapa 5  |	Avaliação dos modelos e otimização              |2 semanas  |
|Etapa 6  |	Desenvolvimento do painel interativo            |2 semanas  |
|Etapa 7  |	Preparação da apresentação final                |1 semana   |


4.	BIBLIOTECAS E REPOSITÓRIO NO GITHUB
Para trabalhar com ciência de dados em Python, utilizaremos as seguintes bibliotecas:
Manipulação de Dados:
●	pandas – Para carregar, limpar e estruturar os dados
●	numpy – Para operações matemáticas e manipulação de arrays

Processamento de Texto (NLP - Natural Language Processing):
![image](https://github.com/user-attachments/assets/6252a56f-d5bf-418d-99f5-0b4be2dbd6aa)


 Visualização de Dados: 
 
![image](https://github.com/user-attachments/assets/3f02540d-5578-46e4-8c20-abfef9960d87)

![image](https://github.com/user-attachments/assets/ad7b4475-a7b0-4c0c-a78f-f99cf0403949)

![image](https://github.com/user-attachments/assets/5a3fbcdf-3a56-4fdf-81c3-03e75e4548fc)


5. Repositório no GitHub:
Criamos um repositório chamado NewsGuardAI-FakeNewsDetections para facilitar a colaboração e controle de versões.
https://github.com/KellyGrazy/NewsGuardAI-FakeNewsDetections

6.	BASE DE DADOS E ANÁLISE EXPLORATÓRIA
Para detectar fake news, utilizaremos bases públicas que contêm notícias classificadas como verdadeiras ou falsas. Algumas opções incluem:
LIAR Dataset: Contém mais de 12.000 declarações classificadas por jornalistas como "verdadeiro", "parcialmente verdadeiro" e "falso".
Fake News Corpus: Uma base extensa com notícias reais e falsas coletadas de diversos sites.
Análise Exploratória:
Carregar os dados no pandas e visualizar as primeiras linhas.
Identificar categorias de notícias (ex: "fake" e "real").
Contar a frequência das classes para verificar balanceamento.
Verificar tamanho médio dos textos e distribuição das palavras mais comuns.

Criar gráficos para entender padrões na distribuição dos textos.
![image](https://github.com/user-attachments/assets/3e1e675d-c97c-4aa7-b1c4-f22bc508b2b4)

 


7.	TRATAMENTO DA BASE DE DADOS
Antes de treinar o modelo, precisamos preparar os dados:
Remover textos duplicados e valores nulos
Converter textos para minúsculas para padronizar
Remover caracteres especiais e stopwords para eliminar ruído
Tokenizar e lematizar palavras (reduzir palavras ao seu radical)
Transformar os textos em vetores numéricos usando TF-IDF ou Word Embeddings (Word2Vec, BERT, etc.)

8.	CÁLCULO DA ACURÁCIA
Para medir o desempenho do modelo, utilizaremos métricas de avaliação de classificação:
Acurácia – Mede a porcentagem de previsões corretas:
Acurácia = Número de previsões corretas
                  Total de previsões feitas 

Precisão e Recall – Avaliam a qualidade das previsões de notícias falsas
F1-Score – Mede o equilíbrio entre precisão e recall
Matriz de Confusão – Para visualizar erros do modelo


Resultados –

 ![image](https://github.com/user-attachments/assets/dcea5a8c-65f2-43a9-8e78-6e62a685ab2e)

  ![image](https://github.com/user-attachments/assets/b51b8865-00b7-4179-b308-e7f4a4cccfcd)



9.	RESULTADOS PRELIMINARES
A partir da metodologia definida na etapa anterior, desenvolvemos modelos de aprendizado supervisionado para classificar notícias como verdadeiros ou falsas. Esse processo seguiu algumas etapas importantes:
•	Pré-processamento dos textos: Começamos limpando os dados, removendo duplicidades e valores nulos. Em seguida, padronizamos os textos, eliminamos palavras irrelevantes e aplicamos a lematização para deixar os termos mais consistentes.
•	Vetorização: Utilizamos a técnica TF-IDF para transformar os textos em representações numéricas.
•	Modelagem: Testamos diferentes algoritmos de classificação.
•	Avaliação dos modelos: Para medir o desempenho dos modelos, usamos métricas como acurácia, precisão, recall, F1-Score e matriz de confusão. 
![image](https://github.com/user-attachments/assets/3b023f1c-f404-4281-a728-017c970bebe0)

10. PRODUTO GERADO E MODELO DE NEGÓCIOS:
Desenvolvemos um protótipo funcional com Streamlit, que oferece uma experiÇencia interativa e intuitiva para análise de notícias. As principais funcionalidades incluem:
•	Entrada de texto (manchete ou notícia completa).
•	Classificação da notícia como “Fake ou Real”, com indicação da probabilidade.
•	Exibição de estatísticas da base de dados utilizada
•	Interface responsiva e fácil de usar.
A NewsGuar AI poderá atuar por meio de diferentes frentes de produtos e serviços:
•	Extensão de navegador: versão gratuita com alertas em tempo real sobre notícias suspeitas.
•	API comercial: venda de acesso à nossa tecnologia de classificação de notícias para desenvolvedores e empresas.
•	Painel analítico corporativo: solução voltada para redações e organizações jornalísticas, com visualizações personalizadas e monitoramento contínuo de notícias.
•	Serviços de consultoria: apoio especializado em checagem automatizada de informações, especialmente útil em campanhas políticas e institucionais.

11. ESBOÇO DO STORYTELLING
Vivemos em um mundo saturado de informações, onde separar o verdadeiro do falso se tornou um desafio diário. Nesse cenário, nasce a NewsGuard AI: uma startup que tem como missão devolver a confiança à informação.
Utilizando inteligência artificial, oferecemos uma ferramenta acessível, eficaz e interativa para que leitores e instituições possam identificar com mais clareza o que a fato e o que é desinformação. Em tempos em que tudo parece incerto, acreditamos que a verdade deve ser o nosso Norte e não pode ser deixada para trás.

12. PROXIMA ETAPA
14.		REFERÊNCIA BIBLIOGRÁFICA
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872.

 
