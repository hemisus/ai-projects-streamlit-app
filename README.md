## AI projects Web Application
지금까지 AI 관련 공부를 하며 예시코드를 따라해보거나 직접 구현한 모델들을 Streamlit을 통해 정리하였습니다.<br>
사용자가 직접 입력값을 넣어 페이지에 구현되어 있는 모델들을 테스트 할 수 있습니다. <br>
각 모델들의 학습된 과정, 데이터 전처리 관련 코드들은 notebooks폴더에 .ipynb파일에서 확인하실 수 있습니다.

### 구현된 기능
- 타이타닉 생존 예측 (ML Classification)
- MNIST 손글씨 숫자 이미지 분류 (CNN)
- 텍스트 감정 분류 (LSTM, KoELECTRA)

### 📁 프로젝트 구조
```text
.
├── artifacts/
├── models/
├── notebooks/
├── pages/
├── utils/
├── app.py
├── requirements.txt
└── README.md
```
### 실행 방법
```bash
git clone https://github.com/hemisus/ai-projects-streamlit-app.git
cd ai-projects-streamlit-app

pip install -r requirements.txt
streamlit run app.py
```

### 사용된 데이터셋
레포지토리에는 데이터셋이 포함되어 있지 않습니다. 아래의 링크에서 데이터를 다운로드 하실 수 있습니다. <br>
Titanic (Kaggle): https://www.kaggle.com/competitions/titanic/data <br>
MNIST: - <br>
NSMC (네이버 영화 리뷰): https://github.com/e9t/nsmc <br>

### 사용된 모델
텍스트 감정분류를 위한 모델로 KoELECTRA 모델이 사용되었습니다. <br>
base: https://huggingface.co/monologg/koelectra-base-v3-discriminator

### 📊 사용 기술
- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Streamlit

### Pytorch 설치 (CUDA 12.1 기준)
CUDA 사용을 위해 아래와 같이 Pytorch를 설치하였습니다. <br>
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### NSMC로 학습된 KoELECTRA모델
https://huggingface.co/Hemisus/koelectra-finetuned-nsmc
### 참고한 자료
딥 러닝 파이토치 교과서 - 입문부터 LLM 파인튜닝까지 (https://wikidocs.net/book/2788) <br>
MNIST CNN: https://wikidocs.net/63618 <br>
LSTM을 이용한 분류: https://wikidocs.net/217687 <br>



