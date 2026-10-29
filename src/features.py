"""Engenharia de features do Titanic.

As funções aqui só criam colunas a partir de informações da própria linha
(nome, família, cabine). Nada é calculado olhando o conjunto inteiro, então
dá para aplicar igual no treino e no teste sem vazamento de dados.

Preencher valores faltantes e transformar texto em número fica para o
Pipeline do scikit-learn, que aprende essas regras só com o treino.
"""

import pandas as pd

# Títulos raros agrupados nos quatro principais
MAPA_TITULOS = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mrs": "Mrs",
    "Mme": "Mrs",
    "Master": "Master",
}

COLUNAS_NUMERICAS = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
COLUNAS_CATEGORICAS = ["Pclass", "Sex", "Embarked", "Title"]
COLUNAS_BINARIAS = ["IsAlone", "HasCabin"]
FEATURES = COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS + COLUNAS_BINARIAS


def extrair_titulo(nomes: pd.Series) -> pd.Series:
    """'Braund, Mr. Owen Harris' -> 'Mr'. Títulos raros viram 'Raro'."""
    titulo = nomes.str.extract(r",\s*([^.]+)\.")[0]
    return titulo.map(MAPA_TITULOS).fillna("Raro")


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma cópia de df com as colunas novas e só as colunas usadas no modelo."""
    df = df.copy()
    df["Title"] = extrair_titulo(df["Name"])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # +1 conta o próprio passageiro
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df[FEATURES]
