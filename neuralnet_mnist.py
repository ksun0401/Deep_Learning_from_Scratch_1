import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 소프트맥스 함수
# 소프츠맥스 함수의 출력은 0 ~ 1.0 사이의 실수이다.
# 또한 출력의 총합은 1이며, 확률로 해석할 수 있다.

def softmax(a):
    c = np.max(a)          # 오버플로를 막기 위해 입력 신호 중 최댓값 사용
    exp_a = np.exp(a - c)  # 지수 함수
    sum_exp_a = np.sum(exp_a)  # 지수 함수의 합
    y = exp_a / sum_exp_a
    return y

# (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)로 반환
# normalize: 0.0 ~ 1.0 사이의 값으로 정규화 할지 정하는 것
# flatten: 1차원 배열로 변환할지 정하는 것
# one_hot_lable: one-hot encoding 형태로 저장할지 정하는 것
# 0 ~ 255 범위인 픽셀의 값들을 0.0 ~ 1.0 으로 정규화
# 기존 1*28*28 3차원에서 784의 1차원으로 변환

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# pickle을 이용해 학습된 가중치 데이터 불러오기
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    # weight와 bias
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 임의의 뉴런(50, 100)을 가진 은닉층 2개
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0

# x에 이미지 데이터 1장 씩 꺼내 predict 함수로 분류
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y)            # argmax로 확률이 가장 높은 원소의 인덱스 구하기
    if p == t[i]:              # 예측 값과 실제 값을 비교, 맞힌 숫자 count
        accuracy_cnt += 1

# 전체 이미지 숫자로 나눠 정확도 구하기
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))   # (Accuracy: 0.9352)
