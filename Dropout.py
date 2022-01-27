import numpy as np

class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_raito = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg = True):
        if train_flg:
            # self.mask에 삭제할 뉴런을 False로 표시한다.
            # self.mask는 x와 형상이 같은 배열로 무작위 생성하고,
            # 그 값이 dropout_ratio보다 큰 원소만 True로 설정한다.
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_raito)

    # 순전파 때 신호를 통과시키는 뉴런은 역전파 때도 신호를 그대로 통과,
    # 순전파 때 통과시키지 않은 뉴런은 역전파 때도 신호를 차단한다.
    def backward(self, dout):
        return dout * self.mask