import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def titanic_preprocess(df):
    df = df.copy()
    # Sex map
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    # Embarked 수동 OHE (항상 컬럼 생성)
    df["Embarked_Q"] = (df["Embarked"] == "Q").astype(int)
    df["Embarked_S"] = (df["Embarked"] == "S").astype(int)

    # Embarked 원본 제거
    df = df.drop("Embarked", axis=1)

    # 필요한 컬럼만 남기기
    final_cols = ['Pclass','Sex','Age','SibSp','Parch','Embarked_Q','Embarked_S']
    df = df[final_cols]

    return df