from tqdm import tqdm  # Progress bar
import dill
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

class FFNN:
  def __init__(self, input_size, hidden_size_array, output_size, activation_function, loss_function, weight_init, regularization=None, lambda_=0.01):
    self.input_size = input_size # 784
    self.hidden_size_array = np.array(hidden_size_array).astype(int) # array jumlah neuron tiap hidden layer
    self.output_size = output_size # 10
    self.num_neurons = np.insert(hidden_size_array, 0, input_size)
    self.num_neurons = np.append(self.num_neurons, output_size).astype(int) # array jumlah neuron termasuk input dan output layer
    self.activation_function = activation_function # array fungsi aktivasi setiap layer (termasuk output)
    self.loss_function = loss_function # hanya untuk output layer. MSE, Binary cross entropy, atau Categorical cross entropy
    self.weight_init = weight_init # array tuple (weight_init, weight_low_or_mean, weight_high_or_std, weight_seed, weight_type) inisialisasi bobot tiap layer (termasuk input)
    self.regularization = regularization # L1, L2, atau None
    self.lambda_ = lambda_ # lambda untuk regularization

    # asumsi model punya minimal satu hidden layer
    self.input_and_hidden_layers = [Layer(self.num_neurons[i], self.num_neurons[i+1], activation_function[i], weight_init[i][0], weight_init[i][1], weight_init[i][2], weight_init[i][3], weight_init[i][4]) for i in range (len(hidden_size_array) + 1)]
    self.output_layer = OutputLayer(output_size, loss_function)

  def forward_propagation(self, data, target):
    # forward propagation satu kali dalam satu batch
    values = data
    weights = [self.input_and_hidden_layers[i].weights for i in range(len(self.num_neurons) - 1)]
    for i in range (len(self.input_and_hidden_layers)):
      values = self.input_and_hidden_layers[i].forward(values)
    self.output_layer.setPredictions(values, target)
    if (self.regularization is not None):
      self.output_layer.calculateLoss(weights, self.regularization, self.lambda_)
    else:
      self.output_layer.calculateLoss()
    return self.output_layer.loss

  def back_propagation(self, learning_rate):
    # backward propagation satu kali dalam satu batch
    # asumsi sudah melakukan forward_propagation sebelum ini
    self.output_layer.lossDerivatives()
    values = self.output_layer.loss_derivatives
    for i in range (len(self.input_and_hidden_layers) - 1, 0, -1):
      values = self.input_and_hidden_layers[i].backward_and_update_weights(values, learning_rate, False)
    self.input_and_hidden_layers[0].backward_and_update_weights(values, learning_rate, True)
    return

  def train_model(self, batch_size, learning_rate, num_epochs, x_train, y_train, x_val, y_val, verbose=0):
    # verbose == 0: tidak menampilkan apa-apa
    # verbose == 1: menampilkan progress bar, kondisi training_loss dan validation_loss
    X_batches_train = np.array_split(x_train, np.ceil(len(x_train) / batch_size))
    Y_batches_train = np.array_split(y_train, np.ceil(len(y_train) / batch_size))
    X_batches_val = np.array_split(x_val, np.ceil(len(x_val) / batch_size))
    Y_batches_val = np.array_split(y_val, np.ceil(len(y_val) / batch_size))
    num_of_batches_train = len(X_batches_train)
    num_of_batches_val = len(X_batches_val)
    training_loss_array = []
    val_loss_array = []
    batches_loss_array = np.array([])
    for i in range (num_epochs):
      progress = range(num_of_batches_train + num_of_batches_val)
      if (verbose == 1): # show progress bar
        progress = tqdm(progress, desc=f"Epoch {i+1}/{num_epochs}", unit="batch")

      batches_loss_array = np.array([])
      for j in range (num_of_batches_train):
        batches_loss_array = np.append(batches_loss_array, self.forward_propagation(X_batches_train[j], Y_batches_train[j])[0].data)
        if (verbose == 1):
          progress.set_postfix({"Batch Loss": batches_loss_array[j]})
          progress.update(1)
        self.back_propagation(learning_rate)
      training_loss_array.append(batches_loss_array.mean())
      batches_loss_array = np.array([])
      for j in range (num_of_batches_val):
        batches_loss_array = np.append(batches_loss_array, self.forward_propagation(X_batches_val[j], Y_batches_val[j])[0].data)
        if (verbose == 1): progress.update(1)
      val_loss_array.append(batches_loss_array.mean())

      if (verbose == 1):
        (print(f"Epoch {i+1}: Train Loss = {training_loss_array[i]}, Val Loss = {val_loss_array[i]}"))
    return training_loss_array, val_loss_array

  def predict(self, x_val):
    predictions = np.empty((0, self.output_size))
    for i in range (len(x_val)):
      values = [x_val[i]]
      for j in range (len(self.input_and_hidden_layers)):
        values = self.input_and_hidden_layers[j].forward(values)
      predictions = np.vstack((predictions, np.vectorize(lambda x: x.data)(values.data)))
    predictions = np.argmax(predictions, axis=1)
    return predictions

  def weight_distribution(self, layers_list):
    # layers_list itu list of integer layer mana saja yang weightnya di plot (mulai dari 0 itu input layer)
    for i in range (len(layers_list)):
      weight_flat = np.vectorize(lambda x: x.data)(self.input_and_hidden_layers[layers_list[i]].weights.data).flatten()
      plt.figure(figsize=(8, 5))
      sns.histplot(weight_flat, bins=50, kde=True, color="blue")
      plt.title("Weight Distribution for layer " + str(layers_list[i]))
      plt.xlabel("Weight Value")
      plt.ylabel("Frequency")
      plt.grid(True)
      plt.show()

  def gradient_distribution(self, layers_list):
    for i in range (len(layers_list)):
      weight_gradient_flat = np.vectorize(lambda x: x.data)(self.input_and_hidden_layers[layers_list[i]].weight_gradients.data).flatten()
      plt.figure(figsize=(8, 5))
      sns.histplot(weight_gradient_flat, bins=50, kde=True, color="blue")
      plt.title("Weight Gradient Distribution for layer " + str(layers_list[i]))
      plt.xlabel("Weight Gradient Value")
      plt.ylabel("Frequency")
      plt.grid(True)
      plt.show()

  def save_model(self, filename):
    with open(filename, "wb") as f:
      dill.dump(self, f)
    print(f"Model saved to {filename}")

  @staticmethod
  def load_model(filename):
    with open(filename, "rb") as f:
      model = dill.load(f)
    print(f"Model loaded from {filename}")
    return model

  def draw_graph(self):
    visualizer = GraphNN(self)
    visualizer.draw_graph()

class GraphNN:
  def __init__(self, model):
    self.layer = model.num_neurons
    self.weights = [model.input_and_hidden_layers[i].weights for i in range(len(self.layer)-1)]
    self.graph = nx.DiGraph()
    self.node_pos = {}
    self.node_lab = {}

    self.build_graph()
  
  def build_graph(self):
    x_offset = 0
    max_neurons = max(self.layer)

    id_node = 0
    prev_layer = []

    for layer_idx, num_neuron in enumerate(self.layer):
      y_offset = (max_neurons - num_neuron) / 2
      current_layer = []

      for i in range(num_neuron):
        if layer_idx > 0 and layer_idx < len(self.layer) - 1:
          label = "h"
        elif layer_idx == len(self.layer) - 1:
          label = "o"
        else: 
          label = "i"
        neuron_label = f"{label}{i+1}"
        self.graph.add_node(id_node, label=neuron_label)
        self.node_pos[id_node] = (x_offset, - i -y_offset)
        self.node_lab[id_node] = neuron_label
        current_layer.append(id_node)
        id_node += 1
      
      if prev_layer:
        weight = self.weights[layer_idx - 1].data
        for j, prev_node in enumerate(prev_layer):
          for k, curr_node in enumerate(current_layer):
            weight_val = weight[j, k].data
            grad_val = weight[j, k].grad
            self.graph.add_edge(prev_node, curr_node, weight=round(weight_val, 2), grad=round(grad_val, 2))

      prev_layer = current_layer
      x_offset += 2

  def draw_graph(self):
    plt.figure(figsize=(10,6))
    edges = self.graph.edges(data=True)

    nx.draw(self.graph, pos=self.node_pos, with_labels=True, labels=self.node_lab, node_size=800, node_color="lightgreen", font_size=10, edge_color="gray")

    edge_labels = {(u, v): f"w={d['weight']}, g={d['grad']}" for u,v,d in edges}
    nx.draw_networkx_edge_labels(self.graph, pos=self.node_pos, edge_labels=edge_labels, font_size=8)

    plt.title("Struktur Jaringan dan Nilai Bobot")
    plt.show()