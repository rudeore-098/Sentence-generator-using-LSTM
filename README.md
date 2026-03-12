# Sentence-generator-using-LSTM
본 프로젝트는 사용자의 입력 문장을 분석하여 문장을 생성하기 위해 **Encoder-Decoder 구조의 Seq2Seq 모델**을 사용합니다.

# Gaol

- LSTM을 활용해 Seq2Seq을 구축
- **Naver sentiment movie**을 이용하여 데이터를 가공
- 아래 조건에 맞는 Input, Output 생성
    - Input <= 10  /  5 <= output <= 20

---

# **Tech Stack**

- **Framework**: `PyTorch`
- **Model Architecture**: `Seq2Seq` (Sequence-to-Sequence)
- **Recurrent Unit**: `LSTM` (Long Short-Term Memory)
- **Optimizer**: `Adam`
- **Loss Function**: `CrossEntropyLoss` (for Token Classification)

---

# parm—

- 단어 사전 크기 | **`INPUT_DIM = len(word2idx)`**
- 생성용 단어 사전 크기 **| `OUTPUT_DIM = len(word2idx)`**
- Encoding 임베딩 차원 | **`ENC_EMB_DIM = 256`**
- Decoding 임베딩 차원 | **`DEC_EMB_DIM = 256`**
- LSTM 은닉 상태 크기 | **`HID_DIM = 512`**
- hidden layer 층 수 | **`N_LAYERS = 2`**
- Encoding 드랍아웃 | **`ENC_DROPOUT = 0.5`**
- Decoding 드랍아웃 | **`DEC_DROPOUT = 0.5`**
- 에포크 | **`N_EPOCHS = 50`**

---

## System Workflow ——

### 1. Preprocessing

- **Tokenization:** `konly` (또는 지정한 토크나이저)를 사용하여 입력 문장을 형태소 단위로 분리.
- **Vectorization -** 학습 시 구축된 `word2idx` 사전을 바탕으로 토큰을 고유 정수 인덱스로 변환.
- **Special Tokens -** 문장의 시작(`<SOS>`)과 끝`<EOS>` 길이를 맞추기 위한 패딩`<PAD>` 처리 그리고 **`<UNK>`** 수행.

### 2. Encoder-Decoder

- 입력된 정수 시퀀스를 **Embedding Layer**를 통해 고차원 벡터로 변환.
- **Multi-layer LSTM**을 거쳐 입력 문장의 전체적인 의미를 함축한 **Context Vector**를 생성.
- `<SOS>` 토큰을 시작으로, 이전 단계에서 출력된 단어를 다음 단계의 입력으로 사용하는 **Autoregressive** 방식으로 단어를 하나씩 예측.

### 3. Post-processing

- `predict_sentence` , `generate_sentence` 을 통해 문장을 생성
- `<EOS>` 토큰이 나오거나 최대 길이에 도달하면 생성을 중단하고 최종 문장을 사용자에게 출력.

---

## 주요 동작 과정 ——

### `def generate_sentence()`

- **Repetition Penalty (중복 억제) -** 동일한 단어가 반복 생성되는 것을 방지하기 위해 생성된 토큰의 Logit 값을 동적으로 차감함.
- **Minimum Length Constraint (최소 길이 보장) -** 설정된 `min_len`에 도달하기 전까지는 `<EOS>` 토큰 생성을 강제로 차감하여 모델이 더 긴 문장을 생성하도록 유도함.
- **Forbidden Token Filtering (금지 토큰 필터링) -** `<UNK>`, `<PAD>`, `<SOS>`와 같이 출력에 부적절한 특수 토큰의 발생 확률을 음수로 만듬.

### `teacher_forcing_ratio`

- Teaching force : teacher_force = random.random() < teacher_forcing_ratio(0.5) 을 통해 50%의 확률로는 정답을 보여주고, 나머지 50%는 모델이 스스로 예측한 토큰을 사용하게 하여 실전(Inference) 적응력을 향상
