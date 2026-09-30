# 텍스트를 임베딩합니다

import os 
import requests
import tiktoken, torch
from torch.utils.data import Dataset, DataLoader

# 사용할 텍스트 가져오기 
if not os.path.exists("the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "the-verdict.txt"

    response = requests.get(url, timeout=30)
    response.raise_for_status() # HTTP 요청이 실패했는지 확인하는 함수, 상태 코드가 오류라면, 예외 발샘
    with open(file_path, "wb") as f:
        f.write(response.content)

# 정규 표현식으로 토큰화하기
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 바이트 페어 인코딩
# 어휘사전에 없는 단어를 더 작은 부분단어나 개별 문자로 분할하여 처리할 수 있습니다. 
tokenizer = tiktoken.get_encoding("gpt2")
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# 슬라이딩 윈도우로 데이터 처리하기
# 텍스트 로드 및 토크나이저 정의
with open("the-verdict.txt", "r",encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
# 텍스트 청크에 대해서 입력과 타겟 
# 타겟은 모델이 다음 단어 예측을 위함. 오른쪽으로 한 토큰 이동한 입력 

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 전체 텍스트를 토큰화
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "토큰화된 입력의 개수는 적어도 max_length + 1 과 같아야합니다."

        # 슬라이딩 윈도를 통해 책을 max_length 길이의 중첩된 시퀀스로 나눕니다.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx],  self.target_ids[idx]

def create_dataloader_v1(txt, batch_size = 4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    # 토크나이저 초기화 
    tokenizer = tiktoken.get_encoding("gpt2")

    # 데이터 셋 만들기
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 데이터 로더 만들기
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# 단어 위치 인코딩하기 
# 바이트페어 인코더의 어휘 사전 크기: 50257
vocab_size = 50257
output_dim = 256

# 토큰 임베딩 층 만들기 
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 배치 크기 = 8, 샘플마다 토큰 4
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride = max_length, shuffle=False
)
data_iter = iter(dataloader)
# input 샘플 하나당  크기 [8,4]
inputs, targets = next(data_iter)

# 임베딩 벡터 값, 토큰 한개당 256 차원, 즉 사이즈는 [8,4,256]
token_embeddings  = token_embedding_layer(inputs)

# 위치 임베딩 층 만들기
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# 위치 임베딩 값 만들기, shape는 [4,256]
pos_embeddings = pos_embedding_layer(torch.arange(max_length))

# 입력 임베딩 = 토큰 임베딩 + 위치 임베딩 
input_embeddings = token_embeddings + pos_embeddings