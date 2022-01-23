import numpy as np
import matplotlib.pyplot as plt


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad


# 경사하강법 구현
# f: 최적화하려는 함수
# init_x: 초깃값
# lr: 학습률(learning rate)
# step_num: 경사법에 따른 반복 횟수

def gradient_descent(f, init_x, lr, step_num):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x) # 함수의 기울기 구하기
        x -= lr * grad    # 기울기에 학습률을 곱한 값으로 초기값 갱신
    return x, np.array(x_history)

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])  # 초깃값 (-3.0, 4.0)
lr = 0.1
step_num = 20

# x, x_history == [-0.03, 0.04]로 (0, 0)에 가까운 결과를 얻었다.
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
