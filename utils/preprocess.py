import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
from PIL import Image
import re

def titanic_preprocess(df):
    df = df.copy()
    # Sex map
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    
    df["Embarked_Q"] = (df["Embarked"] == "Q").astype(int)
    df["Embarked_S"] = (df["Embarked"] == "S").astype(int)
    df = df.drop("Embarked", axis=1)

    final_cols = ['Pclass','Sex','Age','SibSp','Parch','Embarked_Q','Embarked_S']
    df = df[final_cols]

    return df


def preprocess_image_forMNIST(img_array):
    img = Image.fromarray(img_array.astype(np.uint8))
    img = img.resize((28, 28))  #MNIST 데이터는 1x28x28 크기 고정

    img = np.array(img)
    
    img = img / 255.0
    

    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) #(1, 1, 28, 28)로 차원 추가
    return tensor



def clean_text_KOR(text):
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", text)
    return text.strip()