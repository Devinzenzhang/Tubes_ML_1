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
    if isinstance(other, ValueTensor):
      return other + self
    elif isinstance(other, Value):
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
    if isinstance(other, ValueTensor):
      return other + self
    elif isinstance(other, Value):
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
  def lin(self):
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
