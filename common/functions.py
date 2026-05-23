import numpy as np

# 恒等関数
def identity_function(x: np.ndarray) -> np.ndarray:
    return x

# 0以上の場合に1を返して、そうでない場合は0を返すパーセプトロンの活性化関数
def step_function(x: np.ndarray) -> np.ndarray:
    return np.array(x > 0, dtype=int)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# 2乗和誤差（2で割っているのは微分をした時に2が消えるようにするため）
def sum_squared_error(y: np.ndarray, t: np.ndarray) -> float:
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
