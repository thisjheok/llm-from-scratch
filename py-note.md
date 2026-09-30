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