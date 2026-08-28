# ==============================================================================
# PROJETO 6: Indústria 4.0 - Manutenção Preventiva
# Curso: Análise e Desenvolvimento de Sistemas
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. GERAÇÃO DO DATASET DE TREINO E DADOS INÉDITOS
def gerar_dados(n=1000):
    np.random.seed(42)
    temp = np.random.normal(80, 10, n)
    vib = np.random.normal(5, 1.5, n)
    vib[np.random.choice(n, 12)] = 85.0
    vib[np.random.choice(n, 25)] = np.nan
    comb = np.random.choice(['Diesel', 'Eletrico', 'Hibrido'], n)
    rev = np.random.choice(['Sim', 'Não'], n)
    falha_maquina = np.random.choice([0, 1], n, p=[0.5, 0.5])

    df = pd.DataFrame({'temperatura_motor': temp, 'vibracao': vib, 'tipo_combustivel': comb, 'revisao_em_dia': rev, 'falha_maquina': falha_maquina})
    novos = pd.DataFrame({'temperatura_motor': [82.1, 105.0], 'vibracao': [4.8, 12.0], 'tipo_combustivel': ['Eletrico', 'Diesel'], 'revisao_em_dia': ['Sim', 'Não']})
    return df, novos

df, novos_dados = gerar_dados()

# 2. PRÉ-PROCESSAMENTO E TRATAMENTO
df.loc[df["vibracao"] > 20, "vibracao"] = np.nan

X = df.drop(columns=['falha_maquina']).copy()
y = df['falha_maquina']

colunas_num = ['temperatura_motor', 'vibracao']
colunas_bin = ['revisao_em_dia']
colunas_cat = ['tipo_combustivel']

pipeline_num = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessador = ColumnTransformer(
    transformers=[
        ('num', pipeline_num, colunas_num),
        ('bin', OneHotEncoder(drop='if_binary'), colunas_bin),
        ('cat', OneHotEncoder(drop='first'), colunas_cat)
    ]
)

# 3. CONSTRUÇÃO E TREINAMENTO DO MODELO
modelo_pipeline = Pipeline(steps=[
    ('preproc', preprocessador),
    ('modelo', LogisticRegression())
])

modelo_pipeline.fit(X, y)

# 4. INFERÊNCIA E PREDIÇÃO DE PROBABILIDADE
predicoes = modelo_pipeline.predict(novos_dados)
probabilidades = modelo_pipeline.predict_proba(novos_dados)

# 5. EXIBIÇÃO DOS RESULTADOS
print(f"--- RESULTADOS DAS PREDIÇÕES: PROJETO 6 ---")
print("Target Estudado: 'falha_maquina' (0: Normal, 1: Falha)\n")

labels = ['Máquina 1', 'Máquina 2']
for i, linha in novos_dados.iterrows():
    print(f"{labels[i]}: {linha.to_dict()}")
    print(f"   - Predição de falha_maquina: {predicoes[i]}")
    print(f"   - Probabilidade Negativa (0): {probabilidades[i][0]:.2%}")
    print(f"   - Probabilidade Positiva (1): {probabilidades[i][1]:.2%}\n")
