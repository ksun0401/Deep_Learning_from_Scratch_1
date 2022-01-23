import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 가중치, 정규분포로 초기화

    def predict(self, x):   # 예측 수행
        return np.dot(x, self.W) # 행렬곱 연산

    def loss(self, x, t):   # 손실 함수 구하기
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

# x: 입력 데이터, t: 정답 레이블
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

'''
def f(w):
    return net.loss(x, t)
'''
# 위 식을 람다식으로 구현

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)

''' 
w11을 h만큼 늘리면 손실 함수의 값은 0.48h만큼 증가,
w23을 h만큼 늘리면 손실 함수의 값은 0.85h만큼 감소                      
[[ 0.48930244  0.07800316 -0.5673056 ]
 [ 0.73395366  0.11700473 -0.8509584 ]]
'''

