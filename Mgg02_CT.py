torch.cuda.is_available()

# Memuat library yang dibutuhkan dalam pemrosesan data dan pembuatan
# model dengan menggunakan ANN dan SOM berbasis Neural Network
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ==========================================
# 1.1 Persiapan Data
# ==========================================
# Load dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standarisasi Data (Penting untuk ANN agar konvergensi lebih cepat)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Konversi ke PyTorch Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
# Pengecekan ukuran data dari train dan test set
print(f"Ukuran Data Train: {X_train.shape}")
print(f"Ukuran Label Train: {y_train.shape}")
print(f"Ukuran Data Test: {X_test.shape}")
print(f"Ukuran Label Test: {y_test.shape}")

# ==========================================
# 1.2 Definisi Arsitektur Model ANN
# ==========================================
class SimpleANN(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(SimpleANN, self).__init__()
    # Layer 1: Input ke Hidden
    self.layer1 = nn.Linear(input_dim, hidden_dim)
    self.relu = nn.ReLU() # Fungsi Aktivasi
    # Layer 2: Hidden ke Output
    self.layer2 = nn.Linear(hidden_dim, output_dim)
  def forward(self, x):
    out = self.layer1(x)
    out = self.relu(out)
    out = self.layer2(out)
    return out
# Inisialisasi Model
input_dim = 4    # Jumlah fitur Iris
hidden_dim = 10  # Jumlah neuron di hidden layer
output_dim = 3   # Jumlah kelas (Setosa, Versicolor, Virginica)
# Load model untuk digunakan pada pelatihan
model = SimpleANN(input_dim, hidden_dim, output_dim)
print(model)
# Loss Function dan Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 1.3 Proses Pelatihan (Training)
# ==========================================
epochs = 100
loss_history = []
print("Mulai Pelatihan ANN...")
for epoch in range(epochs):
  # Forward pass
  outputs = model(X_train)
  loss = criterion(outputs, y_train)
  # Backward pass dan optimasi
  optimizer.zero_grad() # Reset gradien
  loss.backward()       # Hitung gradien (Backpropagation)
  optimizer.step()      # Update bobot
  loss_history.append(loss.item())
  if (epoch+1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# ==========================================
# 1.4 Evaluasi / Pengujian
# ==========================================
# Plot Loss
plt.plot(loss_history)
plt.title('Grafik Loss Selama Pelatihan ANN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
print("\nEvaluasi pada Data Test...")
with torch.no_grad(): # Matikan gradien untuk testing
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'Akurasi Model: {accuracy * 100:.2f}%')

# ==========================================
# 2.1 Definisi Kelas SOM dengan PyTorch
# ==========================================
class SOM(nn.Module):
  def __init__(self, m, n, dim, n_iter, alpha=None, sigma=None):
    # m, n: Ukuran grid SOM (baris, kolom)
    # dim: Dimensi input data (feature)
    # n_iter: Jumlah iterasi total
    # alpha: Learning rate awal
    # sigma: Radius neighborhood awal
    super(SOM, self).__init__()
    self.m = m
    self.n = n
    self.dim = dim
    self.n_iter = n_iter
    if alpha is None:
      self.alpha = 0.5
    else:
      self.alpha = float(alpha)
    if sigma is None:
      self.sigma = max(m, n) / 2.0
    else:
      self.sigma = float(sigma)

    # Inisialisasi bobot grid secara acak
    self.weights = torch.randn(m * n, dim)

    # Lokasi neuron dalam grid 2D (untuk menghitung jarak tetangga)
    self.locations = torch.tensor(np.array(list(self.neuron_locations())), dtype=torch.float32)

    # Pre-compute peluruhan (decay) learning rate dan sigma
    self.pdist = nn.PairwiseDistance(p=2)
  def neuron_locations(self):
    for i in range(self.m):
      for j in range(self.n):
        yield np.array([i, j])

  def get_bmu(self, x):
    # Mencari Best Matching Unit (BMU)
    # Hitung jarak Euclidean antara input x dan semua neuron
    # x shape: (1, dim), weights shape: (m*n, dim)
    distances = self.pdist(x.unsqueeze(0), self.weights)
    bmu_idx = torch.argmin(distances)
    bmu_loc = self.locations[bmu_idx]
    return bmu_idx, bmu_loc

  def forward(self, x, it):
    # Proses update bobot untuk satu input x pada iterasi ke-it
    bmu_idx, bmu_loc = self.get_bmu(x)

    # Hitung decay parameters
    rate = self.alpha * np.exp(-it / self.n_iter)
    sigma_t = self.sigma * np.exp(-it / self.n_iter)

    # Hitung jarak topologi semua neuron terhadap BMU di grid
    loc_dist = self.pdist(bmu_loc.unsqueeze(0), self.locations)

    # Fungsi Neighborhood (Gaussian)
    influence = torch.exp(- (loc_dist ** 2) / (2 * (sigma_t ** 2)))

    # Update bobot: W_new = W_old + rate * influence * (Input - W_old)
    # Kita perlu reshaping agar dimensi cocok untuk broadcasting
    influence = influence.unsqueeze(1) # shape (m*n, 1)
    x_expanded = x.unsqueeze(0)        # shape (1, dim)
    self.weights += rate * influence * (x_expanded - self.weights)
    return bmu_idx

# ==========================================
# 2.2 Persiapan Data (Tanpa Label)
# ==========================================
iris = load_iris()
data = iris.data
# Normalisasi MinMax (Penting untuk SOM agar jarak Euclidean valid)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data_tensor = torch.tensor(data, dtype=torch.float32)
print(f"Ukuran Data: {data_tensor.shape}")


# ==========================================
# 2.3 Pelatihan SOM
# ==========================================
m, n = 10, 10  # Ukuran Grid 10x10
n_iter = 2000  # Jumlah iterasi
input_dim = data_tensor.shape[1]
som = SOM(m, n, input_dim, n_iter)
print("Mulai Pelatihan SOM...")
for i in range(n_iter):
  # Pilih satu sampel data secara acak
  idx = np.random.randint(0, len(data_tensor))
  x = data_tensor[idx]
  # Lakukan update SOM
  som(x, i)
  if (i+1) % 100 == 0:
    print(f"Iterasi {i+1}/{n_iter} selesai.")

# ==========================================
# 2.4 Visualisasi Hasil (Hit Map dengan Label Asli)
# ==========================================
# Meskipun training unsupervised, akan digunakan label asli untuk melihat
# apakah SOM berhasil memisahkan kelas yang berbeda ke area yang berbeda.
w_x, w_y = zip(*[som.locations[i].numpy() for i in range(m*n)])
mapping = np.zeros((m, n))
plt.figure(figsize=(8, 8))
plt.title("Peta Persebaran Spesies Iris pada Grid SOM")

# Plot neuron grid kosong
plt.xlim(-1, m)
plt.ylim(-1, n)
plt.grid(True)

# Petakan setiap data ke BMU-nya dan beri warna sesuai label asli
colors = ['red', 'green', 'blue']
label_names = ['Setosa', 'Versicolor', 'Virginica']
for i in range(len(data_tensor)):
  x = data_tensor[i]
  y_label = iris.target[i]

  # Cari posisi BMU untuk data ini
  bmu_idx, _ = som.get_bmu(x)
  pos = som.locations[bmu_idx].numpy()

  # Tambahkan sedikit random noise (jitter) agar titik tidak menumpuk
  jitter = np.random.randn(2) * 0.1
  plt.scatter(pos[0] + jitter[0], pos[1] + jitter[1],
              c=colors[y_label], s=50, alpha=0.7)

# Membuat legend manual
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=c, label=l)
                   for c, l in zip(colors, label_names)]
plt.legend(handles=legend_elements, loc='upper right')
plt.show()


def run_ann(hidden_dim=10, lr=0.01, epochs=100, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    class SimpleANN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            return self.layer2(self.relu(self.layer1(x)))

    model = SimpleANN(4, hidden_dim, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_hist = []
    for _ in range(epochs):
        out = model(X_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())

    with torch.no_grad():
        out = model(X_test)
        _, pred = torch.max(out.data, 1)
        acc = (pred == y_test).sum().item() / y_test.size(0)

    return acc, loss_hist


configs = [
    ("baseline", 10, 0.01, 100),
    ("hidden=5",  5,  0.01, 100),
    ("hidden=50", 50, 0.01, 100),
    ("lr=0.1",    10, 0.1,  100),
    ("lr=0.001",  10, 0.001,100),
]

results = []
for name, h, lr, ep in configs:
    acc, loss_hist = run_ann(hidden_dim=h, lr=lr, epochs=ep, seed=42)
    results.append((name, h, lr, ep, acc, loss_hist))
    print(f"{name:10s} | hidden={h:>2} lr={lr:<6} epochs={ep:<3} | acc={acc*100:6.2f}%")


for name, h, lr, ep, acc, loss_hist in results:
    if name in ["baseline", "lr=0.1", "lr=0.001"]:
        plt.plot(loss_hist, label=f"{name} (acc={acc*100:.1f}%)")

plt.title("Perbandingan Loss ANN (baseline vs variasi lr)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


def train_som(m=10, n=10, n_iter=2000, alpha=0.5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    iris = load_iris()
    data = MinMaxScaler().fit_transform(iris.data)
    data_tensor = torch.tensor(data, dtype=torch.float32)

    som = SOM(m, n, dim=data_tensor.shape[1], n_iter=n_iter, alpha=alpha)

    for it in range(n_iter):
        idx = np.random.randint(0, len(data_tensor))
        x = data_tensor[idx]
        som(x, it)

    return som, data_tensor, iris.target

def plot_som(som, data_tensor, labels, title):
    m, n = som.m, som.n
    plt.figure(figsize=(7,7))
    plt.title(title)
    plt.xlim(-1, m)
    plt.ylim(-1, n)
    plt.grid(True)

    colors = ['red','green','blue']
    for i in range(len(data_tensor)):
        x = data_tensor[i]
        y_label = labels[i]
        bmu_idx, _ = som.get_bmu(x)
        pos = som.locations[bmu_idx].numpy()
        jitter = np.random.randn(2) * 0.1
        plt.scatter(pos[0]+jitter[0], pos[1]+jitter[1], c=colors[y_label], s=35, alpha=0.7)
    plt.show()


experiments = [
    ("baseline 10x10 it=2000 a=0.5", 10, 10, 2000, 0.5),
    ("grid 5x5 it=2000 a=0.5",        5,  5,  2000, 0.5),
    ("grid 15x15 it=2000 a=0.5",     15, 15, 2000, 0.5),
    ("iter 500 it=500 a=0.5",        10, 10,  500, 0.5),
    ("alpha 0.9 it=2000 a=0.9",      10, 10, 2000, 0.9),
]

for name, m, n, iters, a in experiments:
    som_e, dt, lbl = train_som(m=m, n=n, n_iter=iters, alpha=a, seed=42)
    plot_som(som_e, dt, lbl, title=name)
