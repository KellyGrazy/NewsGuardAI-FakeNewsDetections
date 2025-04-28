import pandas as pd
import matplotlib.pyplot as plt

# Dados de exemplo (você pode substituir pelos valores reais do seu resultado)
data = {
    "Modelo": [
        "Logistic Regression",
        "Random Forest",
        "Multinomial Naive Bayes",
        "Support Vector Machine (SVM)"
    ],
    "Acurácia": [0.9865, 0.9852, 0.9495, 0.9880],
    "Precisão": [0.9850, 0.9830, 0.9420, 0.9865],
    "Recall": [0.9880, 0.9885, 0.9600, 0.9895],
    "F1-Score": [0.9865, 0.9857, 0.9510, 0.9880]
}

# Criar um DataFrame
df = pd.DataFrame(data)

# Criar a figura
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')

# Criar a tabela
tabela = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(10)
tabela.scale(1.2, 1.2)

# Salvar como imagem
plt.savefig('metrics_summary.png', bbox_inches='tight', dpi=300)
plt.show()
