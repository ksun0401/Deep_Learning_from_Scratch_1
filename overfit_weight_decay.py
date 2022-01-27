import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 데이터를 300개만 사용
x_train = x_train[:300]
t_train = t_train[:300]


# 가중치가 커지는 것을 억제하기 위해
# L2norm을 손실 함수에 더한다.
# L2norm의 람다는 정규화의 세기를 조절하는 하이퍼 파라미터다.
# 람다: 0, 0.1 비교

#weight_decay_lambda = 0.0
weight_decay_lambda = 0.1


network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda)

optimizer = SGD(lr=0.01) # 학습률이 0.01인 SGD로 매개변수 갱신

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


'''
람다가 0.0으로 가중치 감소를 적용하지 않았을 때

epoch:0, train acc:0.09666666666666666, test acc:0.0905
.
.
epoch:199, train acc:1.0, test acc:0.742
epoch:200, train acc:1.0, test acc:0.7425
=======================================


람다가 0.1로 가중치 감소를 적용했을 때

epoch:0, train acc:0.15333333333333332, test acc:0.1277
.
.
epoch:199, train acc:0.9133333333333333, test acc:0.6924
epoch:200, train acc:0.9166666666666666, test acc:0.6926
=======================================

람다가 0일 때는 훈련 데이터의 정확도가 
100에폭을 지나는 무렵부터 거의 100%이다.
그러나 시험 데이터에 대해서는 큰 차이가 보인다.
즉 정확도가 크게 벌어지는 것은 훈련 데이터에만 적응 해버린 결과다.

람다가 0.1일 때는 훈련 데이터와 시험 데이터의 차이가 있지만,
가중치 감소가 적용되지 않았을 때와는 차이가 줄었다.
'''