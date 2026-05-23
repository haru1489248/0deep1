import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.functions import sigmoid, relu, tanh

input_data = np.random.randn(1000, 100) # 1000個のデータ
node_num = 100 # 各隠れ層のノード（ニューロン）の数
hidden_layer_size = 5 # 隠れ層が5層
activations = {} # ここにアクティベーションの結果を格納する

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 初期値
    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

    # Heの初期値（ReLUに適している）
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x, w)

    # 活性化関数
    # z = sigmoid(a)
    # z = relu(a)
    z = tanh(a)

    activations[i] = z

# ヒストグラムを描画
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(-1, 1))
    plt.xticks(np.arange(-1, 1.1, 0.2), rotation=45)
plt.show()
