import matplotlib.pyplot as plt
import numpy as np

# 수치 미분
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 간단한 2차 함수
def function_1(x):
    return 0.01*x**2 + 0.1*x 

# 접선 구하기
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


plt.figure(figsize = (13, 13))
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x", fontsize = 20)
plt.ylabel("f(x)", fontsize = 20)


# x = 5 일때
tf = tangent_line(function_1, 5)
y2 = tf(x)

# x = 10 일때
tf_2 = tangent_line(function_1, 10)
y3 = tf_2(x)

plt.plot(x, y, color = 'black')
plt.plot(x, y2, linestyle = 'dashed', label = 'x = 5', color = 'red')
plt.plot(x, y3, linestyle = 'dotted', label = 'x = 10', color = 'blue')
plt.legend(fontsize = 20)
plt.show()
