import os 
import requests
import re

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

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# 고유한 토큰으로 구성된 단어사전 만들기
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>","<|unk|>"])

vocab = {token:integer for integer, token in enumerate(all_tokens)}

# 단순 토크나이저 구현 
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text