import torch

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