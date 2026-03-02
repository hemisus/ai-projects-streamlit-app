import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocess import clean_text_KOR

LSTM_MODEL_PATH = "models/kor_sentiment_lstm_scripted_model.pt"
ELECTRA_MODEL_PATH ="models/kor_sentiment_koelectra_model"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_lstm = torch.jit.load(LSTM_MODEL_PATH, map_location=device)
model_lstm.eval()
word_to_index = pickle.load(open("artifacts/kor_sentiment_lstm_word_to_index.pkl", "rb"))

tokenizer_electra = AutoTokenizer.from_pretrained(ELECTRA_MODEL_PATH)
model_electra = AutoModelForSequenceClassification.from_pretrained(ELECTRA_MODEL_PATH)
model_electra.to(device)
model_electra.eval()

def predict_lstm(text, model, word_to_index):
    
    stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
    tokens = text.split()
    stopwords_removed_tokens = [word for word in tokens if not word in stopwords] 
    token_indices = [word_to_index.get(token, 1) for token in stopwords_removed_tokens]

    # Convert tokens to tensor
    input_tensor = torch.tensor([token_indices], dtype=torch.long).to(device)  # (1, seq_length)

    # Pass the input tensor through the model
    with torch.no_grad():
        logits = model(input_tensor)  # (1, output_dim)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    # Get the predicted class index
    predicted_index = torch.argmax(logits, dim=1)

    return predicted_index.item(), confidence


def predict_koelectra(text):
    inputs = tokenizer_electra(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model_electra(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return pred, confidence


user_input = st.text_area("문장을 입력하세요")

if st.button("분석하기"):
    if user_input.strip() == "":
        st.warning("문장을 입력해주세요.")
    else:
        clean_text = clean_text_KOR(user_input)
        col1, col2 = st.columns(2)

        # LSTM 예측
        with col1:
            st.subheader("LSTM 결과")
            pred_lstm, conf_lstm = predict_lstm(clean_text, model_lstm, word_to_index)

            if pred_lstm == 1:
                st.success(f"긍정 ({conf_lstm:.2%})")
            else:
                st.error(f"부정 ({conf_lstm:.2%})")

        # ELECTRA 예측
        with col2:
            st.subheader("KoELECTRA 결과")
            pred_ele, conf_ele = predict_koelectra(clean_text)

            if pred_ele == 1:
                st.success(f"긍정 ({conf_ele:.2%})")
            else:
                st.error(f"부정 ({conf_ele:.2%})")