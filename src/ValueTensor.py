import numpy as np

class Value:
  def __init__(self, data, _children=(), _op="", label=""):
    self.data = data               # Data value
    self.grad = 0                  # Grad initialization = 0
    self._backward = lambda: None  # Local backward function
    self._prev = set(_children)    # Previous Values
    self._op = _op                 # Operator
    self.label = label             # Label (variabel name, e.g., x, net, o, h)

  # Data display when printed
  def __repr__(self):
    return f"Value({self.data})"

  # Multiply operator
  def __mul__(self, other):
    # self * other
    # if isinstance(other, ValueTensor):
    #   return other * self
    if isinstance(other, Value):
      other = other
    else:
      other = Value(other)

    out = Value(self.data * other.data, (self, other), "*")

    # Local backpropagation (derivative of out w.r.t self and other)
    def _backward():
      other.grad += self.data * out.grad
      self.grad += other.data * out.grad

    out._backward = _backward  # add _backward function to Value out

    return out

  # Reverse multiply operator
  def __rmul__(self, other):
    # other * self
    return self * other

  # Power operator
  def __pow__(self, other):
    # self**other
    if isinstance(other, (int, float)):
      other = other
    elif isinstance(other, Value):
      other = float(other.data)
    else:
      other = float(other)

    out = Value(self.data**other, (self,), f"**{other}")

    def _backward():
      self.grad += other * (self.data**(other - 1)) * out.grad

    out._backward = _backward

    return out

  def __rpow__(self, other):
    # other**self
    if isinstance(other, (int, float)):
      other = other
    elif isinstance(other, Value):
      other = float(other.data)
    else:
      other = float(other)

    out = Value(other**self.data, (self,), f"{other}**")

    def _backward():
      self.grad += out.data * np.log(other) * out.grad

    out._backward = _backward

    return out

  # Add operator
  def __add__(self, other):
    # self + other
    # if isinstance(other, ValueTensor):
    #   return other + self
    if isinstance(other, Value):
      other = other
    else:
      other = Value(other)

    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
      other.grad += out.grad
      self.grad += out.grad

    out._backward = _backward

    return out

  # reverse add operator
  def __radd__(self, other):
    # other + self
    return self + other

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

  def exp(self):
    out = Value(np.exp(self.data), (self,), "e**")

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out

  def log(self):
    out = Value(np.log(self.data), (self,), "log")

    def _backward():
      self.grad += 1/self.data * out.grad
    out._backward = _backward

    return out

  def abs(self):
    out = Value(np.abs(self.data), (self,), "abs")

    def _backward():
      self.grad += np.sign(self.data) * out.grad
    out._backward = _backward

    return out

  def clip(self, min_val, max_val):
    out_data = np.clip(self.data, min_val, max_val)
    out = Value(out_data, (self,), "clip")

    def _backward():
        if min_val < self.data < max_val:
            self.grad += out.grad
        else:
            self.grad += 0

    out._backward = _backward
    return out

  # Global backward
  def backward(self):
    # Use topological order
    topo = []
    visited = set()
    def build_topo(val):
      if val not in visited:
        visited.add(val)
        for child in val._prev:
          build_topo(child)
        topo.append(val)

    build_topo(self)

    for val in reversed(topo):
      val.grad = 0

    # Set grad to 1 and apply the chain rule
    self.grad = 1
    for val in reversed(topo):
      val._backward()

  # Activation function
  # Linear
  def linear(self):
    out = Value(self.data, (self,), "Linear")

    def _backward():
      self.grad += 1 * out.grad
    out._backward = _backward

    return out

  # ReLU
  def relu(self):
    out = Value(max(0, self.data), (self,), "ReLU")

    def _backward():
      self.grad += (0 if self.data <= 0 else 1) * out.grad
    out._backward = _backward

    return out

  # Sigmoid
  def sigmoid(self):
    out = Value(1/(1 + np.exp(-self.data)), (self,), "Sigmoid")

    def _backward():
      self.grad += out.data * (1 - out.data) * out.grad
    out._backward = _backward

    return out

  # Hyperbolic tangent
  def tanh(self):
    out = Value((np.exp(self.data) - np.exp(-self.data)) \
                / (np.exp(self.data) + np.exp(-self.data)),
                (self,), "tanh")

    def _backward():
      self.grad += (2/(np.exp(self.data) - np.exp(-self.data)))**2 * out.grad
    out._backward = _backward

    return out

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

  def abs(self):
    # abs(self)
    result = np.vectorize(lambda x: x.abs())(self.data)

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
    result = np.vectorize(lambda x: x.linear())(self.data)
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

  def softmax(self):
    out = softmax(self.data)
    return ValueTensor(out)

  def backward(self):
    np.vectorize(lambda x: x.backward())(self.data)

def softmax(values):
    probs = []
    for i in range(len(values)):
      exps = [v.exp() for v in values[i]]
      sum_exps = sum(exps)

      probs_row = [e / sum_exps for e in exps]
      probs.append(probs_row)

    probs = np.array(probs)

    def _make_row_backward(row_idx):
      def _backward():
        probs_row = probs[row_idx]
        out_grad = [v.grad for v in probs_row]

        jac = np.zeros((len(probs_row), len(probs_row)))
        for i, pi in enumerate(probs_row):
            for j, pj in enumerate(probs_row):
                if i == j:
                    d = pi.data * (1 - pi.data)
                else:
                    d = -pi.data * pj.data
                jac[i, j] = d

        grad_input = jac @ out_grad
        for j in range(len(values[row_idx])):
          values[row_idx, j].grad = grad_input[j]
      return _backward

    for row in range(len(probs)):
      row_backward = _make_row_backward(row)
      probs[row, 0]._backward = row_backward

    return probs

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
    t2 = y_true * t1
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
    