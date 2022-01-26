import numpy as np

# 확률적 경사 하강법(SGD)
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr      # lr: 학습률

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]  # 손실 함수의 기울기와 학습률을 곱해, 가중치 갱신

# 모멘텀
class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum =  momentum
        self.v = None  # 물리에서 말하는 속도에 해당

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            # av(momentum*v)는 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

# AdaGrad: 학습을 진행하면서 학습률을 점차 줄여가는 방법
class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key] # h는 기존 기울기 값을 제곱하여 계속 더해준다.
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) # 1e-7은 h가 0이 담겨 있다 해도 0으로 나누는 사태를 막아준다.

# Adam: 모멘텀 + AdaGrad
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1  # 1차 모멘텀용 계수
        self.beta2 = beta2  # 2차 모멘텀용 계수
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


