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

# 덧셈 계층 구현
class AddLayer:
    # 덧셈 계층에서는 초기화가 필요 없으므로 pass를 사용한다.
    def __init__(self):
        pass

    # x, y를 더해서 반환
    def forward(self, x, y):
        out = x + y
        return out

    # 미분 값을 그대로 보낸다.
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy



apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층 생성
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

# 역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price) # 715
print(dapple_num, dapple, dorange, dorange_num, dtax) # 110 2.2 3.3 165 650


