# next(iter)
- next(iter)는 iter에서 다음 값을 하나 꺼내는 python 함수
```python
data_iter = iter([10, 20, 30])

next(data_iter)  # 10
next(data_iter)  # 20
next(data_iter)  # 30
next(data_iter)  # StopIteration 오류
```

# torch.arange(integer)
- torch.arange(max_length)는 0부터 max_length - 1까지의 정수를 담은 PyTorch 텐서를 만든다. 
```python
torch.arange(5)
# tensor([0, 1, 2, 3, 4])
```

# torch.triu
- 텐서에서 대각선 위쪽과 대각선 자체의 값만 남기고, 나머지를 0으로 만드는 함수
```python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
result = torch.triu(x)
# tensor([[1, 2, 3],
#         [0, 5, 6],
#         [0, 0, 9]])
```
- diagonal 인자를 사용하면 기준 대각선을 변경할 수 있다.
-- diagonal=0: 기본값, 주 대각선 포함
-- diagonal=1: 주 대각선 제외, 그 위쪽만 유지
-- diagonal=-1: 주 대각선 아래 한 줄까지 포함