import numpy as np

from Value import Value

# List of value
class ValueTensor:
  def __init__(self, data, label="(h)"):
    if isinstance(data, ValueTensor):
      self.shape = data.shape
      self.dim = data.dim
      self.data = data.data
      self.label = data.label
      return

    if isinstance(data, (list, int, float, Value)):
      data = np.array(data, dtype=object)

    self.shape = data.shape
    self.dim = len(self.shape)
    if self.dim > 0:
      ilabel = np.full(self.shape, np.arange(1, self.shape[-1]+1))
    else:
      ilabel = np.full(self.shape, np.arange(1, 1+1))
    ufunc = np.frompyfunc(lambda val, i: Value(val, label=f"{label}{i}") if not isinstance(val, Value) else val, 2, 1)
    self.data = ufunc(data, ilabel)
    self.label = np.array(np.vectorize(lambda x: x.label)(self.data), dtype=object)

  @property
  def grad(self):
    return np.vectorize(lambda x: x.grad)(self.data)

  @property
  def T(self):
    return ValueTensor(self.data.T)

  def __repr__(self):
    if self.dim > 1:
      return f"ValueTensor(\n{np.vectorize(lambda x: x.data)(self.data)})"
    else:
      return f"ValueTensor({np.vectorize(lambda x: x.data)(self.data)})"

  def __getitem__(self, idx):
    item = self.data[idx]
    if isinstance(item, np.ndarray):
      return ValueTensor(item)
    return item

  def __setitem__(self, idx, val):
    if isinstance(val, (int, float)):
      self.data[idx] = Value(val)

    elif isinstance(val, Value):
      self.data[idx] = val

    elif isinstance(val, (list, np.ndarray)):
      val = np.array(val, dtype=object)
      item = np.vectorize(lambda x: Value(x) if not isinstance(x, Value) else x)(val)
      self.data[idx] = item

    elif isinstance(val, ValueTensor):
      self.data[idx] = val.data

  def append(self, val, axis=0, label="b"):
      if isinstance(val, (int, float)):
          val = Value(val)

      elif isinstance(val, (list, np.ndarray)):
          val = np.array(val, dtype=object)
          val = np.vectorize(lambda x: Value(x) if not isinstance(x, Value) else x)(val)

      elif isinstance(val, ValueTensor):
          val = val.data

      else:
          raise TypeError("Unsupported type for append")

      new_data = np.append(self.data, val, axis=axis)

      return ValueTensor(new_data, label=label)

  def sum(self, axis=0, **kwargs):
    result = np.sum(self.data, axis=axis, **kwargs)
    return ValueTensor(result)

  def mean(self, axis=0, **kwargs):
    result = np.mean(self.data, axis=axis, **kwargs)
    return ValueTensor(result)

  def clip(self, min_val, max_val):
    result = np.vectorize(lambda x: x.clip(min_val, max_val))(self.data)
    return ValueTensor(result)

  # Element wise addition
  def __add__(self, other):
    # self + other
    if isinstance(other, (int, float, Value)):
      if isinstance(other, Value):
        other = other.data
      else:
        other = other
      result = np.vectorize(lambda x: x + other)(self.data)

    elif isinstance(other, ValueTensor):
      if (other.dim == 0):
        result = np.vectorize(lambda x: x + other)(self.data)
      elif self.shape != other.shape:
        raise ValueError("Shapes do not match")
      result = np.vectorize(lambda x, y: x + y)(self.data, other.data)

    elif isinstance(other, (list, np.ndarray)):
      other_tensor = ValueTensor(other)
      return self + other_tensor

    return ValueTensor(result)

  # Element wise reverse addition
  def __radd__(self, other):
    # other + self
    return self + other

  # Element wise multiplication
  def __mul__(self, other):
    # self * other
    if isinstance(other, (int, float, Value)):
      if isinstance(other, Value):
        other = other.data
      else:
        other = other
      result = np.vectorize(lambda x: x * other)(self.data)

    elif isinstance(other, ValueTensor):
      if (other.dim == 0):
        result = np.vectorize(lambda x: x * other)(self.data)
      elif self.shape != other.shape:
        raise ValueError("Shapes do not match")
      result = np.vectorize(lambda x, y: x * y)(self.data, other.data)

    elif isinstance(other, (list, np.ndarray)):
      other_tensor = ValueTensor(other)
      return self * other_tensor

    return ValueTensor(result)

  # Element wise reverse multiplication
  def __rmul__(self, other):
    # other * self
    return self * other

  # Element wise power
  def __pow__(self, other):
    # self**other
    if isinstance(other, (int, float, Value)):
      if not isinstance(other, Value):
        other = Value(other)
      else:
        other = other
      result = np.vectorize(lambda x: x ** other.data)(self.data)

    elif isinstance(other, ValueTensor):
      if self.shape != other.shape:
        raise ValueError("Shapes do not match")
      result = np.vectorize(lambda x, y: x ** y)(self.data, other.data)

    elif isinstance(other, (list, np.ndarray)):
      other_tensor = ValueTensor(other)
      return self ** other_tensor

    return ValueTensor(result)

  # Element wise reverse power
  def __rpow__(self, other):
    # other**self
    if isinstance(other, (int, float, Value)):
      if not isinstance(other, Value):
        other = Value(other)
      else:
        other = other
      result = np.vectorize(lambda x: other.data ** x)(self.data)

    elif isinstance(other, ValueTensor):
      if self.shape != other.shape:
        raise ValueError("Shapes do not match")
      result = np.vectorize(lambda x, y: x ** y)(other.data, self.data)

    elif isinstance(other, (list, np.ndarray)):
      other_tensor = ValueTensor(other)
      return other_tensor ** self

    return ValueTensor(result)

  def exp(self):
    # e**self
    result = np.vectorize(lambda x: x.exp())(self.data)

    return ValueTensor(result)

  def log(self):
    # log(self)
    result = np.vectorize(lambda x: x.log())(self.data)

    return ValueTensor(result)

  def __matmul__(self, other):
    if not isinstance(other, ValueTensor):
        raise TypeError(f"Cannot multiply ValueTensor with {type(other)}")

    if self.shape[-1] != other.shape[0]:
        raise ValueError("Shapes do not match for matrix multiplication")

    result_data = np.empty((self.shape[0], other.shape[1]), dtype=object)

    for i in range(self.shape[0]):
        for j in range(other.shape[1]):
            result_data[i, j] = sum(self.data[i, k] * other.data[k, j] for k in range(self.shape[1]))

    return ValueTensor(result_data)

  def __rmatmul__(self, other):
    if not isinstance(other, ValueTensor):
        raise TypeError(f"Cannot right-multiply ValueTensor with {type(other)}")

    if other.shape[-1] != self.shape[0]:
        raise ValueError("Shapes do not match for matrix multiplication")

    result_data = np.empty((other.shape[0], self.shape[1]), dtype=object)

    for i in range(other.shape[0]):
        for j in range(self.shape[1]):
            result_data[i, j] = sum(other.data[i, k] * self.data[k, j] for k in range(other.shape[1]))

    return ValueTensor(result_data)

  # negative operator
  def __neg__(self):
    # -self
    return self * -1

  # subtract operator
  def __sub__(self, other):
    # self - other
    return self + (-other)

  # reverse subtract operator
  def __rsub__(self, other):
    # other - self
    return other + (-self)

  # Division operator
  def __truediv__(self, other):
    # self / other
    return self * other**(-1)

  # reverse division operator
  def __rtruediv__(self, other):
    # other / self
    return other * self**(-1)

  def linear(self):
    result = np.vectorize(lambda x: x.lin())(self.data)
    return ValueTensor(result)

  def relu(self):
    result = np.vectorize(lambda x: x.relu())(self.data)
    return ValueTensor(result)

  def sigmoid(self):
    result = np.vectorize(lambda x: x.sigmoid())(self.data)
    return ValueTensor(result)

  def tanh(self):
    result = np.vectorize(lambda x: x.tanh())(self.data)
    return ValueTensor(result)

  def softmax(self, axis=-1):
    exp_data = self.exp()
    sum_exp = np.sum(np.vectorize(lambda x: x.data)(exp_data.data), axis=axis, keepdims=True)

    result = np.vectorize(lambda x, s: x / Value(s))(exp_data.data, sum_exp)
    out = ValueTensor(result)

    def _backward():
        soft_vals = np.vectorize(lambda x: x.data)(out.data)
        grad_output = np.vectorize(lambda x: x.grad)(out.data)

        for i in range(soft_vals.shape[0]):
            s = soft_vals[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - (s @ s.T)

            grad_input = jacobian @ grad_output[i].reshape(-1, 1)

            for j in range(soft_vals.shape[1]):
                out.data[i, j].grad += grad_input[j, 0]

    out._backward = _backward

    return out

  def backward(self):
    visited = set()

    def traverse(val):
        if val not in visited:
            visited.add(val)
            for child in val._prev:
                traverse(child)

    for val in np.ravel(self.data):
        traverse(val)

    for val in visited: # set gradien 0
        val.grad = 0

    for val in np.ravel(self.data):
        val.grad = 1

    topo = []
    visited_topo = set()

    def build_topo(val):
        if val not in visited_topo:
            visited_topo.add(val)
            for child in val._prev:
                build_topo(child)
            topo.append(val)

    for val in np.ravel(self.data):
        build_topo(val)

    for val in reversed(topo):
        val._backward()

class criterion:
  # Loss
  # MSE
  def mse(y_true, y_pred):
    y_true = ValueTensor(y_true)
    y_pred = ValueTensor(y_pred)

    mean_ = ((y_true-y_pred)**2).mean(axis=-1)
    mean_ = ValueTensor(np.expand_dims(mean_.data, axis=0))
    return mean_.mean(axis=-1)

  # BCE
  def binary_cross_entropy(y_true, y_pred):
    y_true = ValueTensor(y_true)
    y_pred = ValueTensor(y_pred.clip(1e-10, 1 - 1e-10))

    t1 = y_pred.log()
    t2 = y_true * t1
    t3 = (1 - y_true)
    t4 = (1 - y_pred).log()
    t5 = t3 * t4
    t6 = t2 + t5
    t7 = t6.mean(axis=-1)
    t7 = ValueTensor(np.expand_dims(t7.data, axis=0))
    t8 = t7.mean(axis=-1)
    return -t8

  # CCE
  def categorical_cross_entropy(y_true, y_pred):
    y_true = ValueTensor(y_true)
    y_pred = ValueTensor(y_pred.clip(1e-10, 1 - 1e-10))

    t1 = y_pred.log()
    t2 = y_true *  t1
    t3 = t2.sum(axis=-1)
    t3 = ValueTensor(np.expand_dims(t3.data, axis=0))
    t4 = t3.mean(axis=-1)
    return -t4

  # derivatives
  # output hasil yang belum dikali turunan fungsi aktivasi
  def mse_errors(y_true, y_pred):
    return -2 * (y_true - y_pred) / y_pred.shape[0]

  def bce_errors(y_true, y_pred):
    return -1 * (y_pred - y_true) / (y_pred * (1 - y_pred) * y_pred.shape[0])

  def cce_errors(y_true, y_pred):
    return -1 * y_true / (y_pred * y_pred.shape[0])