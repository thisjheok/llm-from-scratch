# cross entropy
- loss를 구하는 함수 
- 입력: 모델이 추론한 확률 분포 [배치, 단어 사전 크기], 정답 토큰 Index(정답 토큰 인덱스가 1로 되어 있음) : [배치크기]
- loss 를 구하는 과정: 로짓 -> cross entropy에 입력
- pytorch에서 어떻게 구성이 되어있나? (F.cross_entropy())
```python
# 1. Logits -> Log Probabilities
log_probs = F.log_softmax(logits_flat, dim=-1)

# 2. 정답 토큰 위치의 로그 확률만 선택
target_log_probs = log_probs[
    torch.arange(2), targets_flat
]

# 3. 음수 부호 -> 평균
loss = -target_log_probs.mean()
```
- 정답 토큰 Index tensor는 모델이 예측한 확률 중 어떤 확률을 가져와야 하는지 알려주는 역할을 함. 
- Cross Entropy는 logits과 targets의 숫자를 직접 빼서 비교하는 것이 아니다. 정답 토큰 위치의 로그 확률만 선택한다. 
- 정답에 낮은 확률을 부여할 수록, 큰 패널티를 주는 방식으로 학습이 이루어진다.