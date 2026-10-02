import torch
import torch.nn as nn

# 훈련 가능한 가중치가 없는 간소화된 셀프 어텐션 
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[2] #  입력 쿼리 

# 어텐션 점수, w 구하기 
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

# print(attn_scores_2) 
# tensor([0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605]), 입력 원소 6개 => 어텐션 점수 6개 

# 정규화 함수: softmax. 벡터 원소의 합이 1이 되도록 정규화
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 +=  attn_weights_2[i]*x_i

# 모든 입력 토큰에 대해서 어텐션 가중치 계산하기
attn_scores = torch.empty(6,6)

# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i,j] = torch.dot(x_i,x_j)

# 행렬 곱셈을 사용해서 동일한 행렬을 더 효율적으로 구할 수 있다.
attn_scores = inputs @ inputs.T

# softmax 정규화
attn_weights = torch.softmax(attn_scores, dim=-1)

# 문맥 벡터 계산하기
all_context_vecs = attn_weights @ inputs

# 훈련 가능한 가중치를 가진 셀프 어텐션 
x_2 = inputs[1]
d_in = inputs.shape[1] # 입력 임베딩 크기, d=3
d_out = 2 # 출력 임베딩 크기, d=2 

# 1. 가중치 행렬 초기화 
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @  W_key
value_2  = x_2 @ W_value

keys = inputs @ W_key
values = inputs @ W_value

# 정규화 되지 않은 어텐션 점수 
# 2. 쿼리에 대한 모든 어텐션 점수
attn_scores_2 = query_2 @ keys.T

# 3. 소프트맥스 함수를 통해 어텐션 가중치 계산
d_k = keys.shape[1]
# 임베딩 차원의 제곱근으로 나누어 스케일 조정
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)

# 4. 두 번째 입력 쿼리 벡터에 대한 문맥 벡터를 계산
context_vec_2 = attn_weights_2 @ values

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_query
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)

# 가중치 초기화를 nn.Linear로 하기 
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# 코잘 어텐션 마스크 적용하기
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))

masked_simple = attn_weights*mask_simple 
