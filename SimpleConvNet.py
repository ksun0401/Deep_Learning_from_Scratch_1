import numpy as np
from collections import OrderedDict
class SimpleConvNet:
    # conv - relu - pool - affine - relu - affine - softmax

    # 합성곱 계층 하이퍼파라이머 초기화
    # input_dim: 입력 데이터(채널 수, 높이, 너비)의 차원
    # filter_num: 필터 수
    # filter_size: 필터 크기
    # stride: 스트라이드
    # pad: 패딩
    # hidden_size: 은닉층(완전연결) 뉴런 수
    # output_size: 출력층(완전연결) 뉴런 수
    # weight_init_std: 초기화의 가중치 표준편차

    # (30, 1, 5, 5)의 합성곱 계층 가중치 형상
    def __init__(self, input_dim = (1, 28, 28),
                 conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stirde':1},
                 hidden_size = 100, output_size = 10, weight_init_std = 0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['filter_pad']
        filter_stride = conv_param['filter_stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2* filter_pad)/filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))


        #가중치 매개변수 초기화
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['w2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['w3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # CNN 계층 생성
        self.layers = OrderedDict()  # 순서가 있는 딕셔너리
        self.layers['Conv1'] = Convolution(self.params['w1'],
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h = 2, pool_w = 2, stride = 2)
        self.layers['Affine1'] = Affine(self.params['w2'],
                                        self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w3'],
                                        self.params['b3'])
        self.last_layer = SoftmaxWithLoss()


        # 추론 수행을 위한 predict와 loss
        # x: 입력 데이터
        # t: 정답 레이블
        def predict(self, x):
            # 초기화 때 layers에 추가한 계층을 앞에서부터 차례로 forward 메서드 호출하며 결과 다음 계층에 전달
            for layer in self.layers.values():
                x = layer.forward(x)
            return x

        def loss(self, x, t):
            # predict메서드의 결과를 인수로 마지막 층의 forward 메서드 호출
            # 첫 계층 ~ 마지막 계층까지 forward를 처리
            y = self.predict(x)
            return self.last_layer.forward(y, t)

        # 오차역전파법
        def gradient(self, x, t):
            # 순전파
            self.loss(x, t)

            # 역전파
            dout = 1
            dout = self.last_layer.backward(dout)

            layers = list(self.layers.values())
            layers.reverse()
            for layer in layers:
                dout = layer.backward(dout)

            # 결과 저장
            grads = {}
            grads['w1'] = self.layers['Conv1'].dw
            grads['b1'] = self.layers['Conv1'].db
            grads['w2'] = self.layers['Affien1'].dw
            grads['b2'] = self.layers['Affien1'].db
            grads['w3'] = self.layers['Affien2'].dw
            grads['b3'] = self.layers['Affien2'].db

            return grads