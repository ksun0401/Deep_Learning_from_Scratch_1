# AND, NAND, OR, XOR 구현하기
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# AND와는 w와 b의 값만 다르다
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# AND와는 w와 b의 값만 다르다
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print('-'*5,'AND','-'*5)     # -----AND-----
print(AND(0, 0))             # 0
print(AND(1, 0))             # 0
print(AND(0, 1))             # 0
print(AND(1, 1))             # 1


print('-'*5,'NAND','-'*5)    # -----NAND-----
print(NAND(0, 0))            # 1
print(NAND(1, 0))            # 1
print(NAND(0, 1))            # 1
print(NAND(1, 1))            # 0


print('-'*5,'OR','-'*5)      # -----OR-----
print(OR(0, 0))              # 0
print(OR(1, 0))              # 1
print(OR(0, 1))              # 1
print(OR(1, 1))              # 1


# XOR의 경우 단층 퍼셉트론으로는 비선형 영역을 분리할 수 없다.
# 다층 퍼셉트론으로 층을 쌓아 구현한다.

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print('-'*5,'XOR','-'*5)    # -----XOR-----
print(XOR(0, 0))            # 0
print(XOR(1, 0))            # 1
print(XOR(0, 1))            # 1
print(XOR(1, 1))            # 0
