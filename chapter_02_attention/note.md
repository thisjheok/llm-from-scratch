# 셀프 어텐션이란 
- 하나의 입력 시퀀스 내의 서로 다른 위치의 원소 사이에서 어텐션 가중치를 계산 하는 것.
- 각 원소 x^i에 대해서 문맥 벡터 z^i를 계산하는 것이 목표이다.
- z^i는 x^i와 다른 모든 입력(x^1 ~ x^T) 사이에 모든 정보를 담은 임베딩 문맥 벡터이다. 

# 훈련 가능한 가중치가 없는 간단한 셀프 어텐션 
- 쿼리와 x^i 입력을 dot product하여 어텐션 점수 w^i를 구한다. 
- w^i에 대해서 정규화를 거쳐서 어텐션 가중치 a^i를 구한다. (모든 w들의 합이 1이 되도록 하는 등)
- 입력 벡터와 어텐션 가중치를 곱한 후 모두 더하여 i번째 원소에 대한 문맥 벡터 z^i를 구한다.

# nn.Module 객체 호출과 forward
- nn.Module을 상속한 클래스의 경우 객체를 함수처럼 호출할 수 있다.
- mha = MultiHeadAttention 일 경우 mha(batch)가 가능하다.
- 이 호출은 내부적으로 `mha.__call__(batch)`를 거쳐 `forward(batch)`를 실행한다. 따라서 `forward(self, x)`의 `x`에는 호출할 때 넘긴 `batch`가 전달된다

# Multi-Head Attention에서 `transpose`를 하는 이유
```text
(배치, 토큰 수, 헤드 수, 헤드 차원)

1. transpose(1, 2): 헤드별로 토큰을 모으기

queries = queries.transpose(1, 2)
keys = keys.transpose(1, 2)
values = values.transpose(1, 2)

두 축의 순서를 바꾸어 shape을 다음과 같이 만든다.

(배치, 헤드 수, 토큰 수, 헤드 차원)

이렇게 하면 각 배치와 헤드마다 토큰 벡터가 모인다. 따라서 행렬 곱을 통해 헤드별 어텐션을 계산할 수 있다.

2. keys.transpose(2, 3): 쿼리와 키의 토큰별 점수 계산

attn_scores = queries @ keys.transpose(2, 3)

이 시점의 shape은 다음과 같다.

queries:              (배치, 헤드, 쿼리 토큰, 헤드 차원)
keys.transpose(2, 3): (배치, 헤드, 헤드 차원, 키 토큰)

행렬 곱으로 헤드 차원을 맞추면, 각 쿼리 토큰과 키 토큰 조합의 어텐션 점수가 계산된다.

attn_scores: (배치, 헤드, 쿼리 토큰, 키 토큰)
```