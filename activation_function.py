# step, sigmoid 그리고 Relu의 구현과 비교

import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype = int)  # x가 0을 넘으면 1을 출력, 그 외에는 0을 출력

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 0과 1 중 하나만 출력하는 계단 함수와는 달리, 실수(0.3, 0.86 등)을 출력

def relu(x):
    return np.maximum(0, x)  # x가 0을 넘으면 x 그대로 출력, 0 이하이면 0을 출력


x = np.arange(-5.0, 5.0, 0.1)  # x의 범위(-5.0부터 4.9까지 0.1 간격 출력)

# 각 함수의 y값을 생성
step_y = step_function(x)
sigmoid_y = sigmoid(x)
relu_y = relu(x)

# 시각화를 통해 각 함수 비교
plt.figure(figsize = (15, 10))
plt.plot(x, step_y, linestyle = 'dashdot')
plt.plot(x, sigmoid_y, linestyle = 'dotted')
plt.plot(x, relu_y)
plt.ylim(-0.1, 1.1)
plt.show()

