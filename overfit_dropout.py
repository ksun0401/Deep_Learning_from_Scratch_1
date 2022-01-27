import os, sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer # 지금까지 해온 것과 같은 네트워크 학습

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 데이터를 300개만 사용
x_train = x_train[:300]
t_train = t_train[:300]

# 드롭아웃 사용 유무와 비울 설정
use_dropout = False # 드롭아웃을 쓰지 않을 때는 False
dropout_ratio = 0.2


network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list



# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


'''
Dropout = False 일때
=== epoch:301, train acc:1.0, test acc:0.7635 ===
train loss:0.00781566354393583
train loss:0.007410220278648318


Dropout = True 일때

=== epoch:301, train acc:0.7566666666666667, test acc:0.615 ===
train loss:0.847335263150423
train loss:0.6801213486208596


드롭아웃을 적용하지 않았을 때 보다 적용했을 때가
훈련 데이터와 시험 데이터에 대한 정확도 차이가 줄었다.
또한 훈련 데이터에 대한 정확도가 100%에 도달하지 않았다.
'''