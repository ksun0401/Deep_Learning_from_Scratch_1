import numpy as np

# Mean Squared Error(평균 제곱 오차)
# 예측 값과 실제 값의 차(-)를 제곱 후, 그 총합을 구하는 식이다.
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 정답을 원-핫 인코딩으로 표현 정답: 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 예1: '2'일 확률이 높다고 추정
y_1 = [0.1, 0.01, 0.8, 0.0, 0.05, 0.1, 0.0, 0.1, 0.1, 0.0]
# 예2: '7'일 확률이 높다고 추정
y_2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]


# 평균 제곱 오차는 오차가 작을수록 정답에 더 가깝다는 것을 판단한다.
mse_1 = mean_squared_error(np.array(y_1), np.array(t))
mse_2 = mean_squared_error(np.array(y_2), np.array(t))
print(mse_1)  # 0.041
print(mse_2)  # 0.597


# Cross Entropy Error(교차 엔트로피 오차)
# 정답일 때의 추정의 자연로그를 계산하는 식이다.
# log함수에 0을 입력하면 마이너스 무한대를 뜻하는 -inf가 되어 계산을 진행할 수 없게 된다.
# 아주 작은 값(delta)을 더해서 절대 0이 되지 않도록 만든다.
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# 교차 엔트로피 오차도 오차가 작을수록 정답에 더 가깝다는 것을 판단한다.
cross_1 = cross_entropy_error(np.array(y_1), np.array(t))
cross_2 = cross_entropy_error(np.array(y_2), np.array(t))
print(cross_1) # 0.22
print(cross_2) # 2.3