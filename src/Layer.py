class initialization:
  # kelas untuk inisialisasi bobot tiap neuron layer
  # beberapa cara inisialisasi: zero, uniform/normal distribution, xavier/he(bonus)
  # size: tuple (jumlah neuron input, jumlah neuron output)
  def zero(size):
    return np.zeros(size)

  def uniform(size, lower_bound=-1, upper_bound=1, seed=None, method="random"):
    if seed is not None:
      np.random.seed(seed)

    low, high = lower_bound, upper_bound
    if (method == "xavier"):
      x = np.sqrt(6 / (size[0] + size[1]))
      low, high = -x, x
    elif (method == "he"):
      x = np.sqrt(6 / size[0])
      low, high = -x, x

    return np.random.uniform(low=low, high=high, size=size)

  def normal(size, mean=0, std=1, seed=None, method="random"):
    if seed is not None:
      np.random.seed(seed)

    loc, scale = mean, std
    if (method == "xavier"):
      loc, scale = 0, np.sqrt(2 / (size[0] + size[1]))
    elif (method == "he"):
      loc, scale = 0, np.sqrt(2 / size[0])

    return np.random.normal(loc=loc, scale=scale, size=size)
    
class Layer:
  def __init__(self, input_size, output_size, activation_function="linear", weight_init="normal", weight_low_or_mean=None, weight_high_or_std=1, weight_seed=None, weight_type="random"):
    self.input_size = input_size # jumlah neuron di dalam layer ini
    self.output_size = output_size # jumlah neuron di layer selanjutnya. untuk weight
    self.activation_function = activation_function # string??? idk. activation function yg digunakan

    # weight_init (string): zero, uniform, atau normal
    # weight_low_or_mean: untuk lower bound kalau pakai uniform atau mean kalau pakai normal
    # weight_high_or_std: untuk upper bound kalau pakai uniform atau std kalau pakai normal
    # weight_seed: seed untuk inisialisasi weight. untuk reproducibility
    # weight_type (string): random, xavier (bonus), atau he (bonus)
    # inisialisasi weight semua neuron dan bias. size = (input_size + 1, output_size)
    weights_array = None
    if (weight_init == "uniform"):
      if (weight_low_or_mean == None): weights_array = initialization.uniform((input_size + 1, output_size), -1, weight_high_or_std, weight_seed, weight_type)
      else: weights_array = initialization.uniform((input_size + 1, output_size), weight_low_or_mean, weight_high_or_std, weight_seed, weight_type)
    elif (weight_init == "normal"):
      if (weight_low_or_mean == None): weights_array = initialization.normal((input_size + 1, output_size), 0, weight_high_or_std, weight_seed, weight_type)
      else: weights_array = initialization.normal((input_size + 1, output_size), weight_low_or_mean, weight_high_or_std, weight_seed, weight_type)
    else: # weight_init == "zero"
      weights_array = initialization.zero((input_size + 1, output_size))
    self.weights = ValueTensor(weights_array)

    self.neuron_values = None # berisikan semua nilai neuron dan bias dalam satu layer, dalam satu batch??? yang neuronnya sudah dikasih fungsi aktivasi
    self.next_raw = None # untuk simpan data nilai layer selanjutnya yang belum diberi fungsi aktivasi
    self.next_activated = None # untuk simpan data nilai layer selanjutnya yang sudah diberi fungsi aktivasi
    self.next_error = None # untuk simpan gradien untuk backpropagation dan update weight

  def forward(self, inputs): # untuk forward propagation
    if not isinstance(inputs, ValueTensor): inputs = ValueTensor(inputs)

    self.neuron_values = ValueTensor(np.hstack((inputs.data, np.ones((inputs.shape[0], 1))))) # sekalian isiin bias
    self.next_raw = self.neuron_values @ self.weights

    # activation function
    if (self.activation_function == "relu"): self.next_activated = self.next_raw.relu()
    elif (self.activation_function == "sigmoid"): self.next_activated = self.next_raw.sigmoid()
    elif (self.activation_function == "tanh"): self.next_activated = self.next_raw.tanh()
    elif (self.activation_function == "softmax"): self.next_activated = self.next_raw.softmax()
    else: self.next_activated = self.next_raw.linear() # activation function == "linear"

    return self.next_activated

  def backward_and_update_weights(self, next_gradients, learning_rate, is_last): # untuk back propagation dan sekaligus update weights
    # next_gradients itu error layer selanjutnya lagi yg sudah dikaliin dengan weights layer selanjutnya
    # is_last: true kalau bukan yang terakhir
    if not isinstance(next_gradients, ValueTensor): next_gradients = ValueTensor(next_gradients)

    self.next_activated.backward() # untuk self.next_raw.grad

    self.next_error = ValueTensor(self.next_raw.grad) * next_gradients

    if not is_last:
      weight_T_no_bias = ValueTensor(np.array([row[:-1] for row in self.weights.data.T], dtype=object))

    # update weights
    self.weights -= learning_rate * (ValueTensor(self.neuron_values.data.T) @ self.next_error)

    if not is_last: return (self.next_error @ weight_T_no_bias)
    else: return
    
class OutputLayer:
  def __init__(self, output_size, loss_function="mse"):
    self.predicted = None # ValueTensor matriks (batch_size, output_size) prediksi
    self.target = None # target output
    self.loss_function = loss_function # mse, bce, cce
    self.loss = None # nilai loss
    self.loss_derivatives = None # nilai hasil turunan loss yang belum dikali turunan nilai output layer dan sudah dibagi batch size. matriks

  def setPredictions(self, predicted, target):
    if not isinstance(predicted, ValueTensor): self.predicted = ValueTensor(predicted)
    else: self.predicted = predicted
    if not isinstance(target, ValueTensor): self.target = ValueTensor(target)
    else: self.target = target

  def calculateLoss(self):
    # asumsi sudah ada self.predicted dan self.target
    if (self.loss_function == "bce"): self.loss = criterion.binary_cross_entropy(self.target, self.predicted)
    elif (self.loss_function == "cce"): self.loss = criterion.categorical_cross_entropy(self.target, self.predicted)
    else: self.loss = criterion.mse(self.target, self.predicted) # self.loss_function == "mse"

  def lossDerivatives(self):
    # asumsi sudah ada self.predicted dan self.targets
    if (self.loss_function == "bce"): self.loss_derivatives = criterion.bce_errors(self.target, self.predicted)
    elif (self.loss_function == "cce"): self.loss_derivatives = criterion.cce_errors(self.target, self.predicted)
    else: self.loss_derivatives = criterion.mse_errors(self.target, self.predicted) # self.loss_function == "mse"