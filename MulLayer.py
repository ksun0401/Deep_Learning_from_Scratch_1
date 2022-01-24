# 곱셈 계층 구현
class MulLayer:
    def __init__(self):
        # 인스턴스 변수 x, y 초기화
        self.x = None
        self.y = None

    # x, y를 곱하여 반환
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    # 상류에서 넘어온 미분(dout)에 forward 값을 바꿔 곱한다.
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

apple = 100
apple_num = 2
tax = 1.1

# 계층 생성
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price) # 220

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax) # 2.2 110 200


