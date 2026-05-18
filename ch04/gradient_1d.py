import numpy as np
import matplotlib.pyplot as plt

# 中心差分を使用して数値微分を計算する関数
# 中心差分は前方差分、後方差分と比べて誤差が小さい
# 参考： [https://qiita.com/Negelon/items/fafec9b0d33ab96d1446]
def numerical_diff(f, x):
    h = 1e-4 # 0.001
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

# 接線を求める関数
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()

