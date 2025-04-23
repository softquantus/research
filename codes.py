import torch
import torchvision.models as models                                          # ResNet, VGG, etc. 
from transformers import AutoModelForSequenceClassification, AutoTokenizer     # BERT e outros 

import pennylane as qml                                                       # PennyLane QNodes e QNN 

from ipywidgets import interact, FloatSlider                                   # Widgets interativos 
from IPython.display import display                                           # Exibição de widgets


# q8_quantum_sparse_nn.py
import pennylane as qml
import torch
import torch.nn as nn
import torch.quantization as quant

# Seleção do Backend Quântico (mesmo do exemplo anterior)
try:
    from qiskit_aer import Aer
    dev = qml.device("qiskit.aer", wires=4, backend="qasm_simulator")
    print("✅ Running on Qiskit Aer backend.")
except ImportError:
    dev = qml.device("default.qubit", wires=4)
    print("⚠️ Qiskit Aer not found. Using default.qubit simulator.")

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs):
    for i in range(len(inputs)):
        if inputs[i].item() > 0.1:
            qml.RY(inputs[i], wires=i)
            qml.CNOT(wires=[i, (i+1) % len(inputs)])
    return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]

class HybridQuantumSparseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_layer = nn.Linear(4, 4)
        self.activation = nn.Tanh()
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={})
        self.classical_output = nn.Linear(4, 2)

    def forward(self, x):
        x = self.activation(self.classical_layer(x))  # [B, 4]
        outputs = []
        for sample in x:
            outputs.append(self.quantum_layer(sample))
        outputs = torch.stack(outputs)  # [B, 4]
        return self.classical_output(outputs)  # [B, 2]

# Treinamento do modelo (igual ao exemplo anterior)
model = HybridQuantumSparseNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_train = torch.rand(10, 4, device=device)
y_train = torch.randint(0, 2, (10, 2), device=device).float()

for epoch in range(50):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("✅ Training Complete!")

# Após o treinamento, aplique a quantização dinâmica nas camadas clássicas:
model.classical_layer = quant.quantize_dynamic(model.classical_layer, {nn.Linear}, dtype=torch.qint8)
model.classical_output = quant.quantize_dynamic(model.classical_output, {nn.Linear}, dtype=torch.qint8)

print("✅ Quantization Applied!")


# q8+_quantum_sparse_nn.py
import pennylane as qml  # framework de QML usado
import torch
import torch.nn as nn
import torch.ao.quantization as quant

# -------------------------
# 1) Seleção do Backend Quântico
# -------------------------
try:
    from pennylane_lightning_gpu import LightningGPUDevice  # plugin opcional
    dev = qml.device("lightning.gpu", wires=4, diff_method="adjoint")
    print("✅ Running on lightning.gpu backend.")
except ImportError:
    try:
        from qiskit_aer import Aer  # type: ignore
        dev = qml.device("qiskit.aer", wires=4, backend="qasm_simulator")
        print("✅ Running on Qiskit Aer backend.")
    except ImportError:
        dev = qml.device("default.qubit", wires=4)
        print("⚠️ Qiskit Aer not found. Using default.qubit simulator.")  # fallback

# -------------------------
# 2) Definição do QNode (Circuito Quântico)
# -------------------------
@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(inputs):
    """
    Espera inputs de shape [4] (vetor 1D).
    Aplica RY e CNOT apenas quando inputs[i] > 0.1.
    """
    for i in range(len(inputs)):
        if inputs[i] > 0.1:  # tensor escalar, sem .item()
            qml.RY(inputs[i], wires=i)
            qml.CNOT(wires=[i, (i + 1) % len(inputs)])
    return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]

# -------------------------
# 3) Rede Híbrida com Soft‑Threshold e Loop de QNode
# -------------------------
class AdvancedQuantumSparseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_layer = nn.Linear(4, 4)
        self.activation = nn.Tanh()
        self.alpha = nn.Parameter(torch.tensor(10.0))  # parâmetro de "dureza"
        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_circuit, weight_shapes={}
        )
        self.classical_output = nn.Linear(4, 2)

    def forward(self, x):
        # x: [B, 4]
        x = self.activation(self.classical_layer(x))  # [B,4]
        # soft-threshold diferenciado
        mask = torch.sigmoid(self.alpha * (x.abs() - 0.1))  # [B,4]
        x = x * mask
        # envio individual ao QNode
        outputs = []
        for sample in x:  # cada sample: [4]
            outputs.append(self.quantum_layer(sample))
        x = torch.stack(outputs)  # [B,4]
        return self.classical_output(x)  # [B,2]

# -------------------------
# 4) Treinamento
# -------------------------
model = AdvancedQuantumSparseNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Dados de exemplo
X_train = torch.rand(10, 4)
y_train = torch.randint(0, 2, (10, 2)).float()

for epoch in range(50):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("✅ Training Complete!")

# -------------------------
# 5) Quantização Dinâmica
# -------------------------
model.classical_layer = quant.quantize_dynamic(
    model.classical_layer, {nn.Linear}, dtype=torch.qint8
)
model.classical_output = quant.quantize_dynamic(
    model.classical_output, {nn.Linear}, dtype=torch.qint8
)
print("✅ Quantization Applied!")


import pennylane as qml
import torch
import torch.nn as nn
import torch.quantization as quant

# Seleção do Backend Quântico (mesmo do exemplo anterior)
try:
    from qiskit_aer import Aer
    dev = qml.device("qiskit.aer", wires=4, backend="qasm_simulator")
    print("✅ Running on Qiskit Aer backend.")
except ImportError:
    dev = qml.device("default.qubit", wires=4)
    print("⚠️ Qiskit Aer not found. Using default.qubit simulator.")

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs):
    for i in range(len(inputs)):
        if inputs[i].item() > 0.1:
            qml.RY(inputs[i], wires=i)
            qml.CNOT(wires=[i, (i+1) % len(inputs)])
    return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]

class HybridQuantumSparseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_layer = nn.Linear(4, 4)
        self.activation = nn.Tanh()
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={})
        self.classical_output = nn.Linear(4, 2)

    def forward(self, x):
        x = self.activation(self.classical_layer(x))  # [B, 4]
        outputs = []
        for sample in x:
            outputs.append(self.quantum_layer(sample))
        outputs = torch.stack(outputs)  # [B, 4]
        return self.classical_output(outputs)  # [B, 2]

# Treinamento do modelo (igual ao exemplo anterior)
model = HybridQuantumSparseNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_train = torch.rand(10, 4, device=device)
y_train = torch.randint(0, 2, (10, 2), device=device).float()

for epoch in range(50):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("✅ Training Complete!")

# Após o treinamento, aplique a quantização dinâmica nas camadas clássicas:
model.classical_layer = quant.quantize_dynamic(model.classical_layer, {nn.Linear}, dtype=torch.qint8)
model.classical_output = quant.quantize_dynamic(model.classical_output, {nn.Linear}, dtype=torch.qint8)

print("✅ Quantization Applied!")


import matplotlib.pyplot as plt


loss_values = []

for epoch in range(50):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()
    loss_values.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


plt.figure(figsize=(8, 5))
plt.plot(range(1, 51), loss_values, marker='o', label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()


import torch
import torch.nn as nn
import torch.quantization as quant
import matplotlib.pyplot as plt
import pennylane as qml

# Seleção do Backend Quântico
try:
    from qiskit_aer import Aer
    dev = qml.device("qiskit.aer", wires=4, backend="qasm_simulator")
    print("✅ Executando no backend Qiskit Aer.")
except ImportError:
    dev = qml.device("default.qubit", wires=4)
    print("⚠️ Qiskit Aer não encontrado. Usando o simulador default.qubit.")

# Definição do circuito quântico
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs):
    for i in range(len(inputs)):
        if inputs[i].item() > 0.1:
            qml.RY(inputs[i], wires=i)
            qml.CNOT(wires=[i, (i+1) % len(inputs)])
    return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]

# Definição da rede neural híbrida
class HybridQuantumSparseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_layer = nn.Linear(4, 4)
        self.activation = nn.Tanh()
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={})
        self.classical_output = nn.Linear(4, 2)

    def forward(self, x):
        x = self.activation(self.classical_layer(x))  # [B, 4]
        outputs = []
        for sample in x:
            outputs.append(self.quantum_layer(sample))
        outputs = torch.stack(outputs)  # [B, 4]
        return self.classical_output(outputs)  # [B, 2]

# Inicialização do modelo, otimizador e função de perda
model = HybridQuantumSparseNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Definição do dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dados de treinamento fictícios
X_train = torch.rand(10, 4, device=device)
y_train = torch.randint(0, 2, (10, 2), device=device).float()

# Lista para armazenar os valores de perda
loss_values = []

# Loop de treinamento
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()
    loss_values.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("✅ Treinamento concluído!")

# Plotagem da curva de perda
plt.figure(figsize=(8, 5))
plt.plot(range(1, 51), loss_values, marker='o', label='Perda de Treinamento')
plt.title('Curva de Perda durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.grid(True)
plt.legend()
plt.show()

# Aplicação de quantização dinâmica nas camadas clássicas
model.classical_layer = quant.quantize_dynamic(model.classical_layer, {nn.Linear}, dtype=torch.qint8)
model.classical_output = quant.quantize_dynamic(model.classical_output, {nn.Linear}, dtype=torch.qint8)

print("✅ Quantização aplicada!")


import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# For binary classification, select classes 0 and 1
X = X[y < 2]
y = y[y < 2]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the quantum device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# Define the hybrid model
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_net = nn.Linear(4, n_qubits)
        weight_shapes = {"weights": (1, n_qubits)}  # Adjusted to 1 layer
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.post_net = nn.Linear(1, 1)

    def forward(self, x):
        x = self.pre_net(x)
        x = self.q_layer(x)
        x = x.unsqueeze(-1)  # Reshape to (batch_size, 1)
        x = self.post_net(x)
        return torch.sigmoid(x).squeeze()  # Squeeze to match target shape

# Initialize the model, loss function, and optimizer
model = HybridModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 20
loss_list = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Plot the training loss
plt.plot(range(1, epochs + 1), loss_list, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = (predictions >= 0.5).float()
    accuracy = (predicted_classes == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")

import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================
# 1. Configurações Iniciais e Carregamento de Dados
# ==============================================
n_qubits    = 4
n_layers    = 3
n_epochs    = 50
batch_size  = 16
learning_rate = 0.02
n_folds     = 5

iris = load_iris()
X = iris.data[iris.target < 2]
y = iris.target[iris.target < 2]
print("Distribuição de classes:", np.bincount(y))


# ==============================================
# 2. Definição do Circuito Quântico
# ==============================================
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # inputs esperado com len(inputs) == n_qubits
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ==============================================
# 3. Modelo Híbrido
# ==============================================
class QuantumHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        # pré‑rede mapeia 4 features → 4 features
        self.pre_net = nn.Sequential(
            nn.Linear(4, n_qubits),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # camada quântica: n_layers x n_qubits x 3 pesos
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # pós‑rede recebe n_qubits saídas → 1 logit
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        x = self.pre_net(x)              # → [batch, n_qubits]
        x = self.qlayer(x)               # → [batch, n_qubits]
        x = self.post_net(x)             # → [batch, 1]
        return torch.sigmoid(x).squeeze()  # → [batch]


# ==============================================
# 4. Treinamento com Validação Cruzada
# ==============================================
skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/{n_folds}")

    # normalização
    scaler   = StandardScaler()
    X_train  = scaler.fit_transform(X[train_idx])
    X_test   = scaler.transform(X[test_idx])
    X_train  = torch.tensor(X_train, dtype=torch.float32)
    y_train  = torch.tensor(y[train_idx], dtype=torch.float32)
    X_test   = torch.tensor(X_test,  dtype=torch.float32)
    y_test   = torch.tensor(y[test_idx],  dtype=torch.float32)

    model     = QuantumHybrid()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # treino
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss    = nn.BCELoss()(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss.item():.4f}")

    # avaliação
    model.eval()
    with torch.no_grad():
        probs     = model(X_test)
        preds     = (probs >= 0.5).float()
        acc       = (preds == y_test).float().mean().item()
        cm        = confusion_matrix(y_test, preds)
        report    = classification_report(y_test, preds, target_names=iris.target_names[:2])

    results.append({
        'fold': fold+1,
        'accuracy': acc,
        'confusion_matrix': cm,
        'report': report
    })

# ==============================================
# 5. Resultados Agregados
# ==============================================
avg_acc = np.mean([r['accuracy'] for r in results])
print(f"\nAverage Accuracy: {avg_acc:.4f}")

total_cm = np.sum([r['confusion_matrix'] for r in results], axis=0)
plt.figure(figsize=(6,5))
sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names[:2],
            yticklabels=iris.target_names[:2])
plt.title('Matriz de Confusão Agregada')
plt.show()

print("\nRelatório de Classificação (Fold 1):")
print(results[0]['report'])

# salvar modelo final
torch.save(model.state_dict(), 'quantum_hybrid_model.pth')


# Codigo Para Balancemaento de resources
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import threading
import queue
import time
from collections import defaultdict

# 1. Carregamento e Pré-processamento de Dados Genômicos (Simulado)
def load_genomic_data():
    # Gerar dados simulados de SNPs
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    # Simular genótipos: 0=AA, 1=Aa, 2=aa
    data = np.random.choice([0, 1, 2], size=(n_samples, n_features))
    
    # Simular populações (5 classes)
    populations = np.random.choice(['EUR', 'EAS', 'SAS', 'AMR', 'AFR'], size=n_samples)
    
    # Redução de dimensionalidade com PCA
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(data)
    
    return X_pca, pd.factorize(populations)[0]

# 2. Gerenciador de Recursos Híbridos
class HybridResourceManager:
    def __init__(self):
        self.resources = {
            'qpu': [{'id': f'qpu{i}', 'status': 'idle'} for i in range(4)],
            'gpu': [{'id': 'gpu1', 'status': 'idle'}],
            'cpu': [{'id': 'cpu1', 'status': 'idle'}]
        }
        self.lock = threading.Lock()
        self.task_queue = queue.Queue()
        self.failed_tasks = defaultdict(int)
        self.timeout = 10.0

    def allocate_resource(self, task_type):
        with self.lock:
            if task_type == 'quantum':
                for qpu in self.resources['qpu']:
                    if qpu['status'] == 'idle':
                        qpu['status'] = 'busy'
                        return qpu['id']
            else:
                for gpu in self.resources['gpu']:
                    if gpu['status'] == 'idle':
                        gpu['status'] = 'busy'
                        return gpu['id']
                for cpu in self.resources['cpu']:
                    if cpu['status'] == 'idle':
                        cpu['status'] = 'busy'
                        return cpu['id']
            return None

    def release_resource(self, resource_id):
        with self.lock:
            for resource_type in self.resources:
                for resource in self.resources[resource_type]:
                    if resource['id'] == resource_id:
                        resource['status'] = 'idle'
                        break

# 3. Agendador Híbrido
class HybridScheduler:
    def __init__(self):
        self.resource_manager = HybridResourceManager()
        self.worker_threads = []
        self.max_retries = 3
        self._init_workers()

    def _init_workers(self):
        for _ in range(4):
            worker = threading.Thread(target=self.worker_loop, daemon=True)
            worker.start()
            self.worker_threads.append(worker)

    def submit_task(self, task_func, args=(), task_type='quantum'):
        self.resource_manager.task_queue.put((task_func, args, task_type))

    def worker_loop(self):
        while True:
            try:
                task_func, args, task_type = self.resource_manager.task_queue.get(
                    timeout=self.resource_manager.timeout
                )
                resource_id = None
                retry = 0
                
                while retry < self.max_retries:
                    resource_id = self.resource_manager.allocate_resource(task_type)
                    if resource_id:
                        try:
                            task_func(*args, resource_id)
                            break
                        except Exception as e:
                            print(f"Erro na execução: {str(e)}")
                            self.resource_manager.failed_tasks[task_func.__name__] += 1
                            retry += 1
                        finally:
                            if resource_id:
                                self.resource_manager.release_resource(resource_id)
                    else:
                        time.sleep(0.5)
                else:
                    print(f"Tarefa falhou após {self.max_retries} tentativas")
            except queue.Empty:
                break

# 4. Modelo Quântico-Clássico para Genômica
class GenomicQuantumModel(nn.Module):
    def __init__(self, scheduler, n_classes, n_qubits=4):
        super().__init__()
        self.scheduler = scheduler
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        
        # Camada clássica de pré-processamento
        self.classical_encoder = nn.Sequential(
            nn.Linear(n_qubits, 8),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Circuito quântico
        self.quantum_layer = qml.qnn.TorchLayer(self._create_genomic_circuit(), weight_shapes={})
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def _create_genomic_circuit(self):
        dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs):
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
            qml.templates.StronglyEntanglingLayers(
                weights=torch.randn(2, self.n_qubits, 3),
                wires=range(self.n_qubits)
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit

    def forward(self, x):
        x = self.classical_encoder(x)
        
        results = []
        for sample in x:
            self.scheduler.submit_task(
                self._process_quantum_sample,
                (sample.clone().detach(), results),
                'quantum'
            )
        
        start_time = time.time()
        while len(results) < len(x) and (time.time() - start_time) < self.scheduler.resource_manager.timeout:
            time.sleep(0.1)
        
        if len(results) < len(x):
            print(f"Warning: {len(x)-len(results)} amostras não processadas")
            results.extend([torch.zeros(self.n_qubits) for _ in range(len(x)-len(results))])
        
        return self.classifier(torch.stack(results))

    def _process_quantum_sample(self, sample, result_list, resource_id):
        try:
            sample = (sample - sample.mean()) / (sample.std() + 1e-8) * np.pi
            output = self.quantum_layer(sample.unsqueeze(0)).squeeze()
            result_list.append(output)
        except Exception as e:
            print(f"Erro no processamento: {str(e)}")
            result_list.append(torch.zeros(self.n_qubits))

# 5. Pipeline de Treinamento
def train_genomic_model():
    X, y = load_genomic_data()
    n_classes = len(np.unique(y))
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Conversão para tensores
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Inicialização do modelo
    scheduler = HybridScheduler()
    model = GenomicQuantumModel(scheduler, n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.NLLLoss()
    
    # Loop de treinamento
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validação
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            preds = torch.argmax(test_outputs, dim=1)
            accuracy = (preds == y_test).float().mean()
        
        print(f"\nÉpoca {epoch+1}")
        print(f"Loss Treino: {loss.item():.4f} | Loss Teste: {test_loss.item():.4f}")
        print(f"Acurácia: {accuracy.item():.4f}")
        print("Uso de Recursos:", {
            rt: f"{sum(1 for r in res if r['status'] == 'busy')}/{len(res)}"
            for rt, res in scheduler.resource_manager.resources.items()
        })

if __name__ == "__main__":
    train_genomic_model()

# Melhora no Codigo de balaceamento De resource  com dados reais de Genoma e melhoramento de Acurracy
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ==============================================
# 1. Configurações e Carregamento de Dados
# ==============================================
n_qubits = 4
n_layers = 2
n_epochs = 50
learning_rate = 0.01
n_folds = 5

# Carrega dataset Iris (classes 0 e 1)
iris = load_iris()
X = iris.data[iris.target < 2]
y = iris.target[iris.target < 2]
print("Distribuição de classes:", np.bincount(y))

# ==============================================
# 2. Circuito Quântico com Inicialização Correta
# ==============================================
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# ==============================================
# 3. Arquitetura Híbrida Corrigida
# ==============================================
class QuantumHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Camada clássica de pré-processamento
        self.pre_net = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh()
        )
        
        # Camada quântica com inicialização corrigida
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes,
            init_method=lambda _: 0.05*torch.randn(weight_shapes["weights"])
        )
        
        # Camada clássica de pós-processamento
        self.post_net = nn.Linear(1, 1)
    
    def forward(self, x):
        x = self.pre_net(x)
        x = self.qlayer(x)
        x = x.unsqueeze(1)  # Corrige a dimensão para (batch_size, 1)
        x = self.post_net(x)
        return torch.sigmoid(x).squeeze()

# ==============================================
# 4. Pipeline de Treinamento Robustecido
# ==============================================
def train_model():
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{n_folds}")
        
        # Pré-processamento
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X[train_idx]), dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32)
        X_test = torch.tensor(scaler.transform(X[test_idx]), dtype=torch.float32)
        y_test = torch.tensor(y[test_idx], dtype=torch.float32)
        
        # Modelo e otimização
        model = QuantumHybrid()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Treinamento
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # Validação
            if (epoch+1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    preds = (model(X_test) > 0.5).float()
                    acc = (preds == y_test).float().mean().item()
                print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
        
        accuracies.append(acc)
    
    print(f"\nAcurácia média: {np.mean(accuracies):.4f}")
    return model

# ==============================================
# 5. Execução Principal
# ==============================================
if __name__ == "__main__":
    model = train_model()
    torch.save(model.state_dict(), "iris_quantum_model.pth")

# Principais Melhorias Implementadas:

# 📌 Regularização Avançada:
# - Adicionado Dropout(0.3) na camada clássica
# - weight_decay=1e-4 no otimizador (regularização L2)
# - Inicialização Kaiming para pesos das camadas clássicas

# 📊 Monitoramento de Métricas:
# - Precision, Recall e F1-Score calculados para cada fold
# - Matriz de confusão armazenada para o fold problemático (fold 5)
# - Métricas reportadas com média e desvio padrão

# ⏱️ Early Stopping:
# - Interrupção antecipada se não houver melhoria por 10 épocas
# - Sistema de tracking de melhores métricas por fold

# 🧠 Análise do Fold Problemático:
# - Matriz de confusão detalhada para o Fold 5
# - Logs estendidos para identificar problemas específicos

# 🛠️ Melhorias na Arquitetura:
# - Camada clássica expandida: (4 → 8 → 4 neurônios)
# - Inicialização cuidadosa de pesos quânticos e clássicos
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ==============================================
# 1. Configurações e Carregamento de Dados
# ==============================================
n_qubits = 4
n_layers = 2
n_epochs = 100
learning_rate = 0.01
n_folds = 5
patience = 10
random_state = 42

iris = load_iris()
X = iris.data[iris.target < 2]
y = iris.target[iris.target < 2]
print("Distribuição de classes:", np.bincount(y))

# ==============================================
# 2. Circuito Quântico
# ==============================================
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# ==============================================
# 3. Arquitetura Híbrida com Regularização
# ==============================================
class QuantumHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Camada clássica com inicialização Kaiming e dropout
        self.pre_net = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(8, 4),
            nn.Tanh()
        )
        
        # Inicialização dos pesos
        for layer in self.pre_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='tanh')
        
        # Camada quântica
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes,
            init_method=lambda _: nn.init.normal_(torch.empty(weight_shapes["weights"]), 0, 0.1)
        )
        
        # Camada de saída
        self.post_net = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pre_net(x)
        x = self.qlayer(x).unsqueeze(1)
        return self.post_net(x).squeeze()

# ==============================================
# 4. Treinamento com Early Stopping e Métricas
# ==============================================
def train_model():
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'confusion_matrices': []
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{n_folds}")
        
        # Pré-processamento
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X[train_idx]), dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32)
        X_test = torch.tensor(scaler.transform(X[test_idx]), dtype=torch.float32)
        y_test = torch.tensor(y[test_idx], dtype=torch.float32)
        
        # Modelo e otimização
        model = QuantumHybrid()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.BCELoss()
        
        best_loss = float('inf')
        no_improve = 0
        best_metrics = {}

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # Validação
            model.eval()
            with torch.no_grad():
                preds = (model(X_test) > 0.5).float()
                acc = (preds == y_test).float().mean().item()
                
                # Calcula métricas detalhadas
                y_true = y_test.numpy()
                y_pred = preds.numpy()
                
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Early Stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    no_improve = 0
                    best_metrics = {
                        'accuracy': acc,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'epoch': epoch
                    }
                    if fold == 5:  # Salva matriz de confusão para o fold problemático
                        metrics['confusion_matrices'].append(confusion_matrix(y_true, y_pred))
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

        # Atualiza métricas finais do fold
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            metrics[key].append(best_metrics[key])
        
        print(f"Fold {fold} Best - Acc: {best_metrics['accuracy']:.4f} | Prec: {best_metrics['precision']:.4f} | Rec: {best_metrics['recall']:.4f}")

    # Resultados finais
    print("\nMétricas Finais:")
    for metric, values in metrics.items():
        if metric != 'confusion_matrices':
            print(f"{metric.capitalize()} média: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Análise do Fold 5
    if metrics['confusion_matrices']:
        print("\nMatriz de Confusão do Fold 5 (última época):")
        print(metrics['confusion_matrices'][-1])

    return model

# ==============================================
# 5. Execução Principal
# ==============================================
if __name__ == "__main__":
    model = train_model()
    torch.save(model.state_dict(), "improved_iris_quantum_model.pth")

# OpenAI code Solutions 
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# ==============================================
# 1. Configurações Avançadas
# ==============================================
class Config:
    n_qubits      = 4
    n_layers      = 3
    hidden_units  = 8
    dropout_rate  = 0.4
    n_epochs      = 100
    learning_rate = 0.015
    weight_decay  = 1e-3
    patience      = 7
    n_folds       = 5
    random_state  = 42
    tickers       = ["AAPL", "MSFT"]
    start_date    = "2023-01-01"
    end_date      = "2024-04-30"


# ==============================================
# 2. Circuito Quântico Financeiro
# ==============================================
dev = qml.device("default.qubit", wires=Config.n_qubits)

@qml.qnode(dev, interface="torch")
def financial_quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(Config.n_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(Config.n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(Config.n_qubits)]


# ==============================================
# 3. Arquitetura Híbrida Aprimorada
# ==============================================
class FinancialQuantumHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_net = nn.Sequential(
            nn.Linear(4, Config.hidden_units),
            nn.LeakyReLU(0.1),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.hidden_units, Config.n_qubits),
            nn.Tanh(),
        )
        for layer in self.pre_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu", a=0.1)

        weight_shapes = {"weights": (Config.n_layers, Config.n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(
            financial_quantum_circuit,
            weight_shapes,
            init_method=lambda _: nn.init.normal_(
                torch.empty(weight_shapes["weights"]), 0, 0.1
            ),
        )

        self.post_net = nn.Sequential(
            nn.Linear(Config.n_qubits, Config.hidden_units),
            nn.LeakyReLU(0.1),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.pre_net(x)
        x = self.qlayer(x)
        return self.post_net(x).squeeze()


# ==============================================
# 4. Pipeline Completo de Análise
# ==============================================
def main():
    # 4.1 Baixa e processa dados
    print("Baixando dados...")
    df = yf.download(
        Config.tickers,
        start=Config.start_date,
        end=Config.end_date,
        auto_adjust=True,    # 'Close' já ajustado
        # group_by removido → usa padrão ('column')
    )  # :contentReference[oaicite:7]{index=7}

    # 4.2 Extração dos preços de fechamento ajustados
    close_df = df.xs("Close", level=0, axis=1)  # :contentReference[oaicite:8]{index=8}
    adj_close = (
        close_df
        .stack()
        .reset_index()
        .rename(columns={"level_0": "Date", "level_1": "Ticker", 0: "Adj Close"})
    )

    # 4.3 Engenharia de features
    print("\nProcessando features...")
    df_features = adj_close.copy()
    df_features["Return"]     = df_features.groupby("Ticker")["Adj Close"].pct_change()
    df_features["MA10"]       = df_features.groupby("Ticker")["Adj Close"] \
                                         .rolling(10).mean() \
                                         .reset_index(0, drop=True)
    df_features["MA50"]       = df_features.groupby("Ticker")["Adj Close"] \
                                         .rolling(50).mean() \
                                         .reset_index(0, drop=True)
    df_features["Volatility"] = df_features.groupby("Ticker")["Return"] \
                                         .rolling(10).std() \
                                         .reset_index(0, drop=True)
    df_features["Label"]      = df_features.groupby("Ticker")["Return"] \
                                         .shift(-1).gt(0).astype(int)
    df_features = df_features.dropna().reset_index(drop=True)

    X = df_features[["Return","MA10","MA50","Volatility"]].values
    y = df_features["Label"].values

    # 4.4 Validação cruzada
    skf = StratifiedKFold(
        n_splits=Config.n_folds,
        shuffle=True,
        random_state=Config.random_state,
    )
    metrics = {"acc":[], "prec":[], "rec":[], "f1":[], "cms":[], "epochs":[]}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X,y), 1):
        print(f"\n{'='*10} Fold {fold}/{Config.n_folds} {'='*10}")
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X[train_idx]), dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32)
        X_test  = torch.tensor(scaler.transform(X[test_idx]),   dtype=torch.float32)
        y_test  = torch.tensor(y[test_idx], dtype=torch.float32)

        model = FinancialQuantumHybrid()
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate,
                               weight_decay=Config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
        criterion = nn.BCELoss()

        best_loss, no_improve = float("inf"), 0
        best_m = {}

        for epoch in range(Config.n_epochs):
            model.train(); optimizer.zero_grad()
            out = model(X_train)
            loss = criterion(out, y_train)
            loss.backward(); optimizer.step(); scheduler.step(loss)

            model.eval()
            with torch.no_grad():
                prob = model(X_test)
                pred = (prob>=0.5).float()
                val_loss = criterion(prob, y_test)
                acc  = (pred==y_test).float().mean().item()
                prec = precision_score(y_test, pred, zero_division=0)
                rec  = recall_score(y_test, pred, zero_division=0)
                f1   = f1_score(y_test, pred, zero_division=0)

            if val_loss < best_loss:
                best_loss = val_loss
                best_m = {"epoch": epoch+1, "acc":acc, "prec":prec, "rec":rec, "f1":f1,
                          "cm": confusion_matrix(y_test, pred)}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= Config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        metrics["acc"].append(best_m["acc"])
        metrics["prec"].append(best_m["prec"])
        metrics["rec"].append(best_m["rec"])
        metrics["f1"].append(best_m["f1"])
        metrics["cms"].append(best_m["cm"])
        metrics["epochs"].append(best_m["epoch"])

        print(classification_report(y_test, pred))

    # 4.5 Resultados finais
    print("\nMédia de epochs: ", np.mean(metrics["epochs"]))
    print("Acurácia média:  ", np.mean(metrics["acc"]))
    print("F1 média:        ", np.mean(metrics["f1"]))

    return model


# ==============================================
# 5. Execução
# ==============================================
if __name__ == "__main__":
    trained = main()
    torch.save(trained.state_dict(), "financial_quantum_model_v2.pth")
    print("Modelo salvo como 'financial_quantum_model_v2.pth'")


import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             precision_score,
                             recall_score,
                             f1_score)

# ==============================================
# 1. Configurações
# ==============================================
class Config:
    # Parâmetros do Modelo
    n_qubits = 4
    n_layers = 3
    hidden_units = 8
    dropout_rate = 0.4

    # Treinamento
    n_epochs = 100
    learning_rate = 0.015
    weight_decay = 1e-3
    patience = 7
    n_folds = 5
    random_state = 42

    # Dados Financeiros
    tickers = ["AAPL", "MSFT"]
    start_date = "2023-01-01"
    end_date = "2024-04-30"
    use_adjusted_close = True

# ==============================================
# 2. Carregamento de Dados
# ==============================================
def load_financial_data():
    try:
        from yfinance import __version__ as yf_version
        if tuple(map(int, yf_version.split("."))) < (0, 2, 54):
            import pip
            pip.main(["install", "--upgrade", "yfinance"])
    except:
        pass

    df = yf.download(
        tickers=Config.tickers,
        start=Config.start_date,
        end=Config.end_date,
        auto_adjust=Config.use_adjusted_close,
        group_by='ticker'
    )

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Estrutura de colunas inesperada. Atualize o yfinance!")

    column_name = "Close" if Config.use_adjusted_close else "Adj Close"
    try:
        prices = df.xs(column_name, level=1, axis=1)
    except KeyError:
        raise KeyError(f"Coluna '{column_name}' não encontrada. Verifique os parâmetros de download.")

    prices = prices.stack().reset_index()
    prices.columns = ["Date", "Ticker", "Close_Ajustado"]
    return prices

# ==============================================
# 3. Engenharia de Features
# ==============================================
def create_features(prices):
    df = prices.copy()
    df["Return"] = df.groupby("Ticker")["Close_Ajustado"].pct_change()
    df["MA10"] = df.groupby("Ticker")["Close_Ajustado"].rolling(10).mean().reset_index(0, drop=True)
    df["MA50"] = df.groupby("Ticker")["Close_Ajustado"].rolling(50).mean().reset_index(0, drop=True)
    df["Volatility"] = df.groupby("Ticker")["Return"].rolling(10).std().reset_index(0, drop=True)
    df["Label"] = (df.groupby("Ticker")["Return"].shift(-1) > 0).astype(int)
    return df.dropna().reset_index(drop=True)

# ==============================================
# 4. Modelo Quântico
# ==============================================
dev = qml.device("default.qubit", wires=Config.n_qubits)

@qml.qnode(dev, interface="torch")
def financial_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(Config.n_qubits), rotation="Y") 
    qml.StronglyEntanglingLayers(weights, wires=range(Config.n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(Config.n_qubits)]

class QuantumTradingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_net = nn.Sequential(
            nn.Linear(4, Config.hidden_units),
            nn.LeakyReLU(0.1),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.hidden_units, Config.n_qubits),
            nn.Tanh()
        )
        self.qlayer = qml.qnn.TorchLayer(
            financial_circuit,
            {"weights": (Config.n_layers, Config.n_qubits, 3)},
            init_method=lambda _: nn.init.normal_(torch.empty(Config.n_layers, Config.n_qubits, 3), 0, 0.1)
        )
        self.post_net = nn.Sequential(
            nn.Linear(Config.n_qubits, Config.hidden_units),
            nn.LeakyReLU(0.1),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.hidden_units, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pre_net(x)
        x = self.qlayer(x)
        return self.post_net(x).squeeze()

# ==============================================
# 5. Pipeline de Treinamento (Corrigido)
# ==============================================
def main():
    prices = load_financial_data()
    df = create_features(prices)
    X = df[["Return", "MA10", "MA50", "Volatility"]].values
    y = df["Label"].values

    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.random_state)
    metrics = {
        "accuracy": [], 
        "precision": [], 
        "recall": [], 
        "f1": [],
        "confusion_matrices": [], 
        "best_epochs": []
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*40}\nFold {fold}/{Config.n_folds}\n{'='*40}")

        # Pré-processamento
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X[train_idx]), dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32)
        X_test = torch.tensor(scaler.transform(X[test_idx]), dtype=torch.float32)
        y_test = torch.tensor(y[test_idx], dtype=torch.float32)

        # Modelo
        model = QuantumTradingModel()
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
        criterion = nn.BCELoss()

        # Inicialização de métricas
        best_metrics = {
            'epoch': 0,
            'loss': float('inf'),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'cm': np.zeros((2,2))
        }
        no_improve = 0

        for epoch in range(Config.n_epochs):
            # Treino
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # Validação
            model.eval()
            with torch.no_grad():
                probs = model(X_test)
                preds = (probs >= 0.5).float()
                current_metrics = {
                    'loss': criterion(probs, y_test).item(),
                    'accuracy': (preds == y_test).float().mean().item(),
                    'precision': precision_score(y_test, preds, zero_division=0),
                    'recall': recall_score(y_test, preds, zero_division=0),
                    'f1': f1_score(y_test, preds, zero_division=0),
                    'cm': confusion_matrix(y_test, preds)
                }

            # Atualiza melhores métricas
            if current_metrics['loss'] < best_metrics['loss']:
                best_metrics = {
                    'epoch': epoch + 1,
                    **current_metrics
                }
                no_improve = 0
            else:
                no_improve += 1

            # Early stopping
            if no_improve >= Config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            # Log
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Val Loss: {current_metrics['loss']:.4f}")
                print(f"Accuracy: {current_metrics['accuracy']:.4f} | Precision: {current_metrics['precision']:.4f}")
                print(f"Recall: {current_metrics['recall']:.4f} | F1: {current_metrics['f1']:.4f}")

        # Armazena resultados
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            metrics[key].append(best_metrics[key])
        metrics['confusion_matrices'].append(best_metrics['cm'])
        metrics['best_epochs'].append(best_metrics['epoch'])

        print(f"\nMelhores resultados Fold {fold}:")
        print(f"Época: {best_metrics['epoch']} | Loss: {best_metrics['loss']:.4f}")
        print(classification_report(y_test, preds, target_names=["Queda", "Alta"]))

    # Resultados finais
    print("\nResultados Finais:")
    print(f"Acurácia: {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
    print(f"Precisão: {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
    print(f"Recall: {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")
    print(f"F1-Score: {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")

    return model

# ==============================================
# 6. Execução
# ==============================================
if __name__ == "__main__":
    trained_model = main()
    torch.save(trained_model.state_dict(), "quantum_trading_model_final.pth")
    print("\nModelo salvo com sucesso!")



pip install yfinance torch pennylane numpy pandas scikit-learn

import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

# ==============================================
# 1. Configurações Atualizadas
# ==============================================
class Config:
    # Parâmetros Quânticos
    n_qubits = 4
    n_layers = 3

    # Rede Neural
    # <-- Ajustado para 6 features: Return, RSI, MACD, MA10, MA50, Volatility
    input_features = 6
    hidden_units = 8
    dropout_rate = 0.3

    # Treinamento
    n_epochs = 100
    learning_rate = 0.01
    weight_decay = 1e-3
    patience = 10
    n_folds = 5
    random_state = 42

    # Dados
    tickers = ["AAPL", "MSFT"]
    start_date = "2020-01-01"
    end_date = "2024-04-30"
    rolling_window = 14
    use_adjusted_close = True
    class_weights = True

# ... (funções calculate_rsi, calculate_macd e load_and_process_data continuam iguais)

# ==============================================
# 4. Arquitetura Quântica Corrigida
# ==============================================
dev = qml.device("default.qubit", wires=Config.n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(Config.n_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(Config.n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(Config.n_qubits)]

class QuantumTradingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_net = nn.Sequential(
            nn.Linear(Config.input_features, Config.hidden_units),
            nn.ReLU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.hidden_units, Config.n_qubits),
            nn.Tanh()
        )
        weight_shape = (Config.n_layers, Config.n_qubits, 3)
        self.qlayer = qml.qnn.TorchLayer(
            quantum_circuit,
            {"weights": weight_shape},
            init_method=lambda _: nn.init.normal_(torch.empty(*weight_shape), 0, 0.1)
        )
        self.post_net = nn.Sequential(
            nn.Linear(Config.n_qubits, Config.hidden_units),
            nn.ReLU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.hidden_units, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pre_net(x)
        x = self.qlayer(x)
        return self.post_net(x).squeeze()

# ==============================================
# 5. Pipeline de Treino Ajustado
# ==============================================
# ==============================================
# 5. Pipeline de Treino Ajustado
# ==============================================
def train_model():
    df = load_and_process_data()
    X = df.drop("Label", axis=1).values
    y = df["Label"].values

    if Config.class_weights:
        cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
        class_weights = torch.tensor(cw, dtype=torch.float32)
    else:
        class_weights = None

    skf = StratifiedKFold(
        n_splits=Config.n_folds,
        shuffle=True,
        random_state=Config.random_state
    )

    # ← Aqui, uso 'acc','prec','rec','f1' para bater com best_metrics
    metrics = {
        "acc": [],
        "prec": [],
        "rec": [],
        "f1": [],
        "confusion_matrices": []
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'-'*40}\nFold {fold}/{Config.n_folds}\n{'-'*40}")

        scaler   = StandardScaler()
        X_train  = torch.tensor(scaler.fit_transform(X[train_idx]), dtype=torch.float32)
        y_train  = torch.tensor(y[train_idx], dtype=torch.float32)
        X_test   = torch.tensor(scaler.transform(X[test_idx]), dtype=torch.float32)
        y_test   = torch.tensor(y[test_idx], dtype=torch.float32)

        model    = QuantumTradingModel()
        optimizer= optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        criterion= nn.BCELoss()

        best_metrics = {"loss": float("inf")}
        no_improve   = 0

        for epoch in range(Config.n_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss    = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                probs    = model(X_test)
                preds    = (probs >= 0.5).float()
                val_loss = criterion(probs, y_test)
                acc      = (preds == y_test).float().mean().item()
                prec     = precision_score(y_test, preds, zero_division=0)
                rec      = recall_score(y_test, preds, zero_division=0)
                f1       = f1_score(y_test, preds, zero_division=0)

            if val_loss.item() < best_metrics["loss"]:
                best_metrics = {
                    "loss": val_loss.item(),
                    "acc": acc,
                    "prec": prec,
                    "rec": rec,
                    "f1": f1,
                    "cm": confusion_matrix(y_test, preds)
                }
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:03d} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {val_loss.item():.4f}"
                )
                print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

            if no_improve >= Config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # ← Agora essas chaves existem em metrics
        metrics["acc"].append(best_metrics["acc"])
        metrics["prec"].append(best_metrics["prec"])
        metrics["rec"].append(best_metrics["rec"])
        metrics["f1"].append(best_metrics["f1"])
        metrics["confusion_matrices"].append(best_metrics["cm"])

        print(f"\nMelhores Resultados Fold {fold}:")
        print(classification_report(y_test, preds, target_names=["Queda", "Alta"]))

    print("\nResultados Finais:")
    print(f"Acurácia: {np.mean(metrics['acc']):.4f} ± {np.std(metrics['acc']):.4f}")
    print(f"F1-Score: {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")

    return model


# ==============================================
# 6. Execução
# ==============================================
if __name__ == "__main__":
    trained = train_model()
    torch.save(trained.state_dict(), "quantum_trader_final.pth")
    print("Modelo treinado e salvo com sucesso!")


!pip install pennylane torch scikit-learn imbalanced-learn matplotlib


# quantum_med_classifier.py
import os
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, confusion_matrix, 
                             roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE

class QuantumMedClassifier:
    def __init__(self, config=None):
        # Configurações padrão para dados médicos
        self.config = {
            'n_qubits': 6,
            'n_layers': 3,
            'n_epochs': 100,
            'learning_rate': 0.005,
            'patience': 15,
            'n_folds': 5,
            'random_state': 42,
            'use_smote': True,
            'batch_size': 32,
            'output_dir': 'medical_models/'
        }
        if config: 
            self.config.update(config)
            
        # Criar diretório de saída
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        self._init_quantum_device()
        self.scaler = StandardScaler()
        self.model = None
        self.metrics = {
            'auc': [], 'f1': [], 'precision': [], 'recall': [],
            'confusion_matrices': [], 'roc_curves': []
        }

    def _init_quantum_device(self):
        self.dev = qml.device("default.qubit", wires=self.config['n_qubits'])
        
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(
                inputs, 
                wires=range(self.config['n_qubits']), 
                rotation='Y'
            )
            qml.BasicEntanglerLayers(
                weights, 
                wires=range(self.config['n_qubits'])
            )
            return qml.expval(qml.PauliZ(0))
        
        self.circuit = circuit

    def _build_model(self, input_dim):
        class MedicalModel(nn.Module):
            def __init__(self, config, quantum_circuit):
                super().__init__()
                self.config = config
                self.quantum_circuit = quantum_circuit
                
                # Pré-processamento clássico
                self.classical_net = nn.Sequential(
                    nn.Linear(input_dim, 16),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(0.3),
                    nn.Linear(16, self.config['n_qubits']),
                    nn.Tanh()
                )
                
                # Camada quântica
                self.quantum_layer = qml.qnn.TorchLayer(
                    self.quantum_circuit,
                    {"weights": (self.config['n_layers'], self.config['n_qubits'])},
                    init_method=nn.init.uniform_
                )
                
                # Pós-processamento
                self.post_net = nn.Sequential(
                    nn.Linear(1, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                x = self.classical_net(x)
                x = self.quantum_layer(x).unsqueeze(1)
                return self.post_net(x).squeeze()
        
        return MedicalModel(self.config, self.circuit)

    def fit(self, X, y):
        self._validate_input(X, y)
        self._preprocess_data(X, y)
        self._train_model()
        return self

    def _validate_input(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Número de amostras em X e y não correspondem")
        if len(np.unique(y)) < 2:
            raise ValueError("Dados devem conter pelo menos duas classes")

    def _preprocess_data(self, X, y):
        if self.config['use_smote']:
            X, y = SMOTE(random_state=self.config['random_state']).fit_resample(X, y)
        
        self.scaler.fit(X)
        self.X = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.model = self._build_model(X.shape[1])

    def _train_model(self):
        skf = StratifiedKFold(n_splits=self.config['n_folds'], shuffle=True,
                             random_state=self.config['random_state'])
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y), 1):
            self._train_single_fold(fold, train_idx, val_idx)
            self._save_metrics(fold, val_idx)

    def _train_single_fold(self, fold, train_idx, val_idx):
        model = self._build_model(self.X.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=self.config['learning_rate'],
                                    weight_decay=1e-4)
        
        best_auc = 0
        no_improve = 0
        best_metrics = {}
        
        # Garantir que o diretório existe
        os.makedirs(self.config['output_dir'], exist_ok=True)

        for epoch in range(self.config['n_epochs']):
            # Fase de treino
            model.train()
            optimizer.zero_grad()
            outputs = model(self.X[train_idx])
            loss = nn.BCELoss()(outputs, self.y[train_idx])
            loss.backward()
            optimizer.step()
            
            # Fase de validação
            model.eval()
            with torch.no_grad():
                probs = model(self.X[val_idx])
                y_pred = (probs > 0.5).float()
                
                # Métricas robustas com tratamento de divisão zero
                try:
                    auc = roc_auc_score(self.y[val_idx].numpy(), probs.numpy())
                    precision = precision_score(self.y[val_idx].numpy(), y_pred.numpy(), zero_division=0)
                    recall = recall_score(self.y[val_idx].numpy(), y_pred.numpy(), zero_division=0)
                    f1 = f1_score(self.y[val_idx].numpy(), y_pred.numpy(), zero_division=0)
                except ValueError:
                    auc, precision, recall, f1 = 0, 0, 0, 0

            # Early stopping e salvamento
            if auc > best_auc:
                best_auc = auc
                no_improve = 0
                best_metrics = {
                    'auc': auc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'model_state': model.state_dict()
                }
                torch.save(model.state_dict(), 
                          f"{self.config['output_dir']}/best_fold{fold}.pt")
            else:
                no_improve += 1

            if no_improve >= self.config['patience']:
                break

            # Logging aprimorado
            if epoch % 10 == 0:
                print(f"Fold {fold} | Época {epoch+1:03d} | "
                      f"Loss: {loss.item():.4f} | AUC: {auc:.4f} | "
                      f"Precision: {precision:.2f} | Recall: {recall:.2f}")

        # Carregar melhor modelo do fold
        model.load_state_dict(best_metrics['model_state'])
        self.model = model

    def _save_metrics(self, fold, val_idx):
        with torch.no_grad():
            probs = self.model(self.X[val_idx]).numpy()
            y_true = self.y[val_idx].numpy()
        
        # Métricas com tratamento de erros
        try:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc = roc_auc_score(y_true, probs)
            cm = confusion_matrix(y_true, (probs > 0.5))
        except Exception as e:
            print(f"Erro ao calcular métricas: {str(e)}")
            return

        self.metrics['roc_curves'].append((fpr, tpr))
        self.metrics['auc'].append(auc)
        self.metrics['f1'].append(f1_score(y_true, (probs > 0.5), zero_division=0))
        self.metrics['precision'].append(precision_score(y_true, (probs > 0.5), zero_division=0))
        self.metrics['recall'].append(recall_score(y_true, (probs > 0.5), zero_division=0))
        self.metrics['confusion_matrices'].append(cm)

    def predict(self, X):
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return (self.model(X) > 0.5).float().numpy()

    def get_performance_report(self):
        return {
            'mean_auc': np.mean(self.metrics['auc']),
            'std_auc': np.std(self.metrics['auc']),
            'mean_f1': np.mean(self.metrics['f1']),
            'mean_precision': np.mean(self.metrics['precision']),
            'mean_recall': np.mean(self.metrics['recall']),
            'confusion_matrices': self.metrics['confusion_matrices']
        }

    def save_model(self, filename='quantum_med_model.pt'):
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config
        }, filename)

    @classmethod
    def load_model(cls, filename):
        checkpoint = torch.load(filename)
        model = cls(checkpoint['config'])
        model.scaler = checkpoint['scaler']
        model.model = model._build_model(model.scaler.n_features_in_)
        model.model.load_state_dict(checkpoint['model_state'])
        return model

if __name__ == "__main__":
    # Exemplo com dados de câncer de mama
    from sklearn.datasets import load_breast_cancer
    
    # Carregar dados
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Configuração otimizada
    config = {
        'n_qubits': 8,
        'n_layers': 4,
        'n_epochs': 150,
        'learning_rate': 0.001,
        'patience': 20,
        'output_dir': 'breast_cancer_models/'
    }
    
    # Garantir que o diretório existe
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Treinar e avaliar
    print("Iniciando treinamento do modelo quântico...")
    model = QuantumMedClassifier(config).fit(X, y)
    
    # Resultados
    report = model.get_performance_report()
    print("\n=== Relatório de Performance ===")
    print(f"AUC Médio: {report['mean_auc']:.2%} (±{report['std_auc']:.2%})")
    print(f"Precisão Média: {report['mean_precision']:.2%}")
    print(f"Recall Médio: {report['mean_recall']:.2%}")
    print(f"F1-Score Médio: {report['mean_f1']:.2%}")
    
    # Salvar modelo
    model.save_model('modelo_cancer_mama.pt')
    print("\nModelo treinado e salvo com sucesso!")

# ✅ 3. Para Produto Real: Adapte para Casos de Uso
# Transforme este pipeline em uma biblioteca plugável.

# 🧠 Troque o dataset (Iris) por dados reais de diagnóstico:
# - Substitua o dataset Iris por datasets clínicos (ex: câncer, EEG, ECG, etc.)
# - Com ajustes mínimos, o modelo pode ser adaptado para:
#     • Classificação de células
#     • Análise preditiva de doenças
#     • Avaliação de risco em tempo real

# 💡 Ação recomendada:
# - Crie uma interface para que seu time ou seus clientes possam trocar `X` e `y` facilmente.
# - Permita treinar o modelo híbrido com apenas **1 linha de código**.
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, f1_score, 
                             precision_score, recall_score,
                             confusion_matrix, roc_curve,
                             precision_recall_curve)  # Importação corrigida
from imblearn.over_sampling import SMOTE
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class QuantumConfig:
    def __init__(self):
        self.n_qubits = 4
        self.n_layers = 3
        self.ranges = [1] * 3
        self.measurements = 2
        self.noise_level = 0.01

class QuantumClinicalModel(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config
        
        self.classical_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, config.n_qubits),
            nn.Tanh()
        )
        
        self.dev = qml.device("default.qubit", wires=config.n_qubits)
        
        @qml.qnode(self.dev, interface='torch')
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(config.n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(
                weights=weights,
                wires=range(config.n_qubits),
                ranges=config.ranges
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(config.measurements)]
        
        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_circuit,
            {"weights": (config.n_layers, config.n_qubits, 3)},
            init_method=nn.init.kaiming_normal_
        )
        
        self.post_net = nn.Sequential(
            nn.Linear(config.measurements, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.classical_net(x)
        x = self.quantum_layer(x)
        return self.post_net(x).squeeze()

class QuantumTrainer:
    def __init__(self, config):
        self.config = config
        self.scaler = RobustScaler()
        self.best_metrics = {'auc': 0, 'f1': 0}
        self.threshold = 0.5

    def prepare_data(self):
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        X, y = SMOTE().fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )

    def optimize_threshold(self, probs, y_true):
        precisions, recalls, thresholds = precision_recall_curve(y_true.numpy(), probs.numpy())
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        self.threshold = thresholds[np.argmax(f1_scores)]

    def train(self):
        X_train, y_train, X_test, y_test = self.prepare_data()
        
        model = QuantumClinicalModel(X_train.shape[1], self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=200)
        criterion = nn.BCELoss()
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), 
                                batch_size=32, shuffle=True)
        
        for epoch in range(100):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                probs = model(X_test)
                self.optimize_threshold(probs, y_test)
                preds = (probs >= self.threshold).float()
                
                auc = roc_auc_score(y_test.numpy(), probs.numpy())
                f1 = f1_score(y_test.numpy(), preds.numpy())
                
                if auc > self.best_metrics['auc']:
                    self.best_metrics = {'auc': auc, 'f1': f1}
                    torch.save(model.state_dict(), 'best_model.pth')
                
                print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss/len(train_loader):.4f} | "
                      f"AUC: {auc:.4f} | F1: {f1:.4f}")

        self.plot_performance(X_test, y_test)

    def plot_performance(self, X_test, y_test):
        model = QuantumClinicalModel(X_test.shape[1], self.config)
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        
        with torch.no_grad():
            probs = model(X_test).numpy()
            fpr, tpr, _ = roc_curve(y_test, probs)
            
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, probs):.2f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Taxa de Falsos Positivos')
            plt.ylabel('Taxa de Verdadeiros Positivos')
            plt.title('Curva ROC - Modelo Quântico')
            plt.legend()
            plt.savefig('curva_roc.png')
            plt.close()

if __name__ == "__main__":
    config = QuantumConfig()
    trainer = QuantumTrainer(config)
    trainer.train()
    
    print("\n=== Resultados Finais ===")
    print(f"AUC: {trainer.best_metrics['auc']:.2%}")
    print(f"F1-Score: {trainer.best_metrics['f1']:.2%}")

# 🚀 Principais Melhorias no Pipeline de Avaliação:

# ✅ Validação Cruzada Robusta:
# - 5-fold estratificado para melhor representatividade dos dados
# - Relatórios individuais por fold
# - Métricas consolidadas (média e desvio padrão entre folds)

# 🔍 Monitoramento de Overfitting:
# - Curvas de aprendizado (loss de treino vs validação)
# - Early stopping implícito com base na estabilização das curvas
# - Análise de variância entre folds para medir estabilidade do modelo

# 🧬 Métricas Clínicas Relevantes:
# - Sensibilidade (Recall) para identificar casos positivos
# - Especificidade para evitar falsos positivos
# - Curvas Precision-Recall para avaliação de modelos desequilibrados
# - Matrizes de confusão detalhadas para cada fold

# 📊 Visualizações Detalhadas:
# - Curvas ROC por fold para medir capacidade discriminativa
# - Curvas de aprendizado (train/val loss ao longo das épocas)
# - Relatório agregado com insights quantitativos e qualitativos
import os
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, f1_score, 
                             precision_score, recall_score,
                             confusion_matrix, roc_curve,
                             precision_recall_curve)
from imblearn.over_sampling import SMOTE
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset

class QuantumClinicalModel(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config
        
        # Rede neural clássica
        self.classical_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LayerNorm(64),
            nn.Linear(64, config['n_qubits']),
            nn.Tanh()
        )
        
        # Circuito quântico
        self.dev = qml.device("default.qubit", wires=config['n_qubits'])
        
        @qml.qnode(self.dev, interface='torch')
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(config['n_qubits']), rotation='Y')
            qml.StronglyEntanglingLayers(
                weights=weights,
                wires=range(config['n_qubits']),
                ranges=config['ranges']
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(config['measurements'])]
        
        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_circuit,
            {"weights": (config['n_layers'], config['n_qubits'], 3)},
            init_method=nn.init.kaiming_normal_
        )
        
        # Camada de saída
        self.post_net = nn.Sequential(
            nn.Linear(config['measurements'], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.classical_net(x)
        x = self.quantum_layer(x)
        return self.post_net(x).squeeze()

class MedicalQuantumTrainer:
    def __init__(self, config):
        self.config = config
        self.scaler = RobustScaler()
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def load_data(self):
        data = load_breast_cancer()
        X, y = data.data, data.target
        return X, y

    def preprocess_data(self, X, y):
        X, y = SMOTE().fit_resample(X, y)
        X = self.scaler.fit_transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def train_model(self):
        X, y = self.load_data()
        X_tensor, y_tensor = self.preprocess_data(X, y)
        
        skf = StratifiedKFold(n_splits=self.config['n_folds'], shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tensor, y_tensor)):
            print(f"\n=== Fold {fold+1}/{self.config['n_folds']} ===")
            
            model = QuantumClinicalModel(X.shape[1], self.config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['lr'], weight_decay=1e-3)
            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=200)
            criterion = nn.BCELoss()
            
            train_loader = DataLoader(TensorDataset(X_tensor[train_idx], y_tensor[train_idx]), 
                                    batch_size=32, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_tensor[val_idx], y_tensor[val_idx]), 
                                  batch_size=32)
            
            best_val_auc = 0
            train_losses, val_losses = [], []
            
            for epoch in range(self.config['epochs']):
                # Fase de treino
                model.train()
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item()
                
                # Fase de validação
                model.eval()
                val_loss = 0
                all_probs, all_labels = [], []
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        probs = model(X_val)
                        val_loss += criterion(probs, y_val).item()
                        all_probs.extend(probs.numpy())
                        all_labels.extend(y_val.numpy())
                
                # Cálculo de métricas
                train_losses.append(epoch_loss/len(train_loader))
                val_losses.append(val_loss/len(val_loader))
                auc = roc_auc_score(all_labels, all_probs)
                
                # Early stopping e salvamento
                if auc > best_val_auc:
                    best_val_auc = auc
                    torch.save(model.state_dict(), f"{self.config['output_dir']}/best_fold{fold}.pth")
                
                # Plotagem em tempo real
                if epoch % 10 == 0:
                    self._plot_learning_curves(train_losses, val_losses, fold)
                    print(f"Epoch {epoch+1:03d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | AUC: {auc:.4f}")
            
            # Avaliação final do fold
            fold_metrics.append(self._evaluate_fold(model, val_loader, all_labels))
        
        # Análise final
        self._analyze_results(fold_metrics)
    
    def _evaluate_fold(self, model, val_loader, y_true):
        model.eval()
        with torch.no_grad():
            probs = torch.cat([model(X_val) for X_val, _ in val_loader])
        
        # Métricas clínicas
        fpr, tpr, _ = roc_curve(y_true, probs)
        prec, rec, _ = precision_recall_curve(y_true, probs)
        cm = confusion_matrix(y_true, (probs >= 0.5).int())
        
        return {
            'auc': roc_auc_score(y_true, probs),
            'f1': f1_score(y_true, (probs >= 0.5).int()),
            'sensitivity': recall_score(y_true, (probs >= 0.5).int()),
            'specificity': cm[0,0]/(cm[0,0]+cm[0,1]),
            'roc_curve': (fpr, tpr),
            'pr_curve': (prec, rec),
            'confusion_matrix': cm
        }
    
    def _plot_learning_curves(self, train_loss, val_loss, fold):
        plt.figure(figsize=(10,6))
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.title(f'Fold {fold+1} - Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{self.config['output_dir']}/learning_curves_fold{fold}.png")
        plt.close()
    
    def _analyze_results(self, fold_metrics):
        print("\n=== Análise Final ===")
        print(f"AUC Médio: {np.mean([m['auc'] for m in fold_metrics]):.2%} (±{np.std([m['auc'] for m in fold_metrics]):.2%})")
        print(f"Sensibilidade Média: {np.mean([m['sensitivity'] for m in fold_metrics]):.2%}")
        print(f"Especificidade Média: {np.mean([m['specificity'] for m in fold_metrics]):.2%}")
        
        # Plotagem agregada
        plt.figure(figsize=(10,6))
        for i, metrics in enumerate(fold_metrics):
            plt.plot(*metrics['roc_curve'], label=f'Fold {i+1} (AUC={metrics["auc"]:.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curvas ROC por Fold')
        plt.legend()
        plt.savefig(f"{self.config['output_dir']}/aggregated_roc.png")
        plt.close()

if __name__ == "__main__":
    config = {
        'n_qubits': 4,
        'n_layers': 3,
        'measurements': 2,
        'ranges': [1, 1, 1],  # Um range por camada
        'n_folds': 5,
        'epochs': 100,
        'lr': 0.001,
        'output_dir': 'quantum_medical_reports'
    }
    
    trainer = MedicalQuantumTrainer(config)
    trainer.train_model()

    print("\n=== Resultados Finais ===")
    print(f"AUC: {trainer.best_metrics['auc']:.2%}")
    print(f"F1-Score: {trainer.best_metrics['f1']:.2%}")



import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, f1_score, 
                           precision_score, recall_score,
                           confusion_matrix, roc_curve, 
                           accuracy_score)
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset
import os

# Configurações
fin_config = {
    'n_qubits': 4,
    'n_layers': 3,
    'ranges': [1, 1, 1],
    'measurements': 2,
    'seq_length': 30,
    'features': ['Close', 'Volume', 'RSI', 'MACD'],
    'ticker': 'AAPL',
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'test_size': 0.2,
    'batch_size': 64,
    'epochs': 50,
    'lr': 0.001,
    'output_dir': 'quantum_finance_reports'
}

# Criar diretório de saída se não existir
os.makedirs(fin_config['output_dir'], exist_ok=True)

class FinancialQuantumModel(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config
        
        self.classical_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.LayerNorm(64),
            nn.Linear(64, config['n_qubits']),
            nn.Tanh()
        )
        
        self.dev = qml.device("default.qubit", wires=config['n_qubits'])
        
        @qml.qnode(self.dev, interface='torch')
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(config['n_qubits']), rotation='Y')
            qml.StronglyEntanglingLayers(
                weights=weights,
                wires=range(config['n_qubits']),
                ranges=config['ranges']
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(config['measurements'])]
        
        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_circuit,
            {"weights": (config['n_layers'], config['n_qubits'], 3)},
            init_method=nn.init.kaiming_uniform_
        )
        
        self.post_net = nn.Sequential(
            nn.Linear(config['measurements'], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.classical_net(x)
        x = self.quantum_layer(x)
        return self.post_net(x).squeeze(-1)  # Corrigido para garantir dimensão correta

class QuantumFinanceAnalyzer:
    def __init__(self, config):
        self.config = config
        self.scaler = RobustScaler()
        self.X = None
        self.y = None
        
    def fetch_and_preprocess(self):
        print("Baixando dados do Yahoo Finance...")
        df = yf.download(
            self.config['ticker'],
            start=self.config['start_date'],
            end=self.config['end_date'],
            progress=False
        )
        
        print("Calculando indicadores técnicos...")
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['MACD'] = self._calculate_macd(df['Close'])
        df = df.dropna()
        
        close_prices = df['Close'].values
        features = df[self.config['features']].values
        
        print("Criando sequências temporais...")
        sequences, labels = [], []
        for i in range(len(df) - self.config['seq_length'] - 1):
            seq = features[i:i+self.config['seq_length']]
            current_idx = i + self.config['seq_length']
            next_idx = current_idx + 1
            label = 1 if close_prices[next_idx] > close_prices[current_idx] else 0
            sequences.append(seq)
            labels.append(label)
            
        self.X = np.array(sequences)
        self.y = np.array(labels)
        
        print("Normalizando dados...")
        n_samples, seq_len, n_features = self.X.shape
        self.X = self.scaler.fit_transform(self.X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
        
        print(f"\nDados pré-processados:")
        print(f"- Total de amostras: {len(self.X)}")
        print(f"- Proporção de labels positivas: {np.mean(self.y):.2%}")
        print(f"- Shape dos dados: {self.X.shape}\n")
        
    def _calculate_rsi(self, series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, series, slow=26, fast=12, signal=9):
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.ewm(span=signal).mean()
    
    def train_model(self):
        print("Preparando dados para treinamento...")
        split_idx = int(len(self.X) * (1 - self.config['test_size']))
        X_train, X_test = self.X[:split_idx], self.X[split_idx:]
        y_train, y_test = self.y[:split_idx], self.y[split_idx:]
        
        print("\nDivisão dos dados:")
        print(f"- Treino: {len(X_train)} amostras")
        print(f"- Teste: {len(X_test)} amostras")
        print(f"- Proporção positiva no treino: {np.mean(y_train):.2%}")
        print(f"- Proporção positiva no teste: {np.mean(y_test):.2%}\n")
        
        model = FinancialQuantumModel(X_train.shape[1] * X_train.shape[2], fin_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['lr'], weight_decay=1e-3)
        criterion = nn.BCELoss()
        
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train.reshape(-1, X_train.shape[1]*X_train.shape[2]), dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            ),
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        print("Iniciando treinamento...\n")
        best_f1 = 0
        for epoch in range(self.config['epochs']):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Avaliação
            model.eval()
            with torch.no_grad():
                test_probs = model(torch.tensor(X_test.reshape(-1, X_test.shape[1]*X_test.shape[2]), dtype=torch.float32))
                test_preds = (test_probs > 0.5).float()
                
                # Métricas
                auc = roc_auc_score(y_test, test_probs.numpy())
                f1 = f1_score(y_test, test_preds.numpy())
                accuracy = accuracy_score(y_test, test_preds.numpy())
                precision = precision_score(y_test, test_preds.numpy())
                recall = recall_score(y_test, test_preds.numpy())
                
                # Atualiza melhor modelo
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), f"{self.config['output_dir']}/best_model.pth")
                
                print(f"Epoch {epoch+1:03d}/{self.config['epochs']} | Loss: {total_loss/len(train_loader):.4f}")
                print(f"  AUC: {auc:.2%} | F1: {f1:.2%} | Accuracy: {accuracy:.2%}")
                print(f"  Precision: {precision:.2%} | Recall: {recall:.2%}\n")
        
        # Carrega melhor modelo
        model.load_state_dict(torch.load(f"{self.config['output_dir']}/best_model.pth"))
        print(f"\nMelhor F1-score alcançado: {best_f1:.2%}")
        
        # Avaliação final
        self.evaluate_model(model, X_test, y_test)
        self.plot_performance(model, X_test, y_test)
    
    def evaluate_model(self, model, X_test, y_test):
        print("\nAvaliação Final do Modelo:")
        model.eval()
        with torch.no_grad():
            test_probs = model(torch.tensor(X_test.reshape(-1, X_test.shape[1]*X_test.shape[2]), dtype=torch.float32))
            test_preds = (test_probs > 0.5).float()
            
            # Métricas
            auc = roc_auc_score(y_test, test_probs.numpy())
            f1 = f1_score(y_test, test_preds.numpy())
            accuracy = accuracy_score(y_test, test_preds.numpy())
            precision = precision_score(y_test, test_preds.numpy())
            recall = recall_score(y_test, test_preds.numpy())
            cm = confusion_matrix(y_test, test_preds.numpy())
            
            print(f"- AUC: {auc:.4f}")
            print(f"- F1-score: {f1:.4f}")
            print(f"- Accuracy: {accuracy:.4f}")
            print(f"- Precision: {precision:.4f}")
            print(f"- Recall: {recall:.4f}")
            print("\nMatriz de Confusão:")
            print(cm)
            
            # Salva métricas em arquivo
            with open(f"{self.config['output_dir']}/metrics.txt", "w") as f:
                f.write(f"AUC: {auc:.4f}\n")
                f.write(f"F1-score: {f1:.4f}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write("\nMatriz de Confusão:\n")
                f.write(str(cm))
    
    def plot_performance(self, model, X_test, y_test):
        model.eval()
        with torch.no_grad():
            probs = model(torch.tensor(X_test.reshape(-1, X_test.shape[1]*X_test.shape[2]), dtype=torch.float32))
            
            # Curva ROC
            plt.figure(figsize=(10,6))
            fpr, tpr, _ = roc_curve(y_test, probs)
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, probs):.2%}')
            plt.plot([0,1], [0,1], 'k--')
            plt.title('Curva ROC - Desempenho do Modelo')
            plt.xlabel('Taxa de Falsos Positivos')
            plt.ylabel('Taxa de Verdadeiros Positivos')
            plt.legend()
            plt.savefig(f"{self.config['output_dir']}/roc_curve.png")
            plt.close()
            
            # Sinais de Trading
            plt.figure(figsize=(12,6))
            plt.plot(probs.numpy(), label='Probabilidade de Alta', alpha=0.7)
            plt.plot(y_test, 'g--', label='Real', alpha=0.5)
            plt.title('Sinais de Trading vs Realidade')
            plt.xlabel('Amostras')
            plt.ylabel('Probabilidade/Classe Real')
            plt.legend()
            plt.savefig(f"{self.config['output_dir']}/trading_signals.png")
            plt.close()
            
            # Distribuição das Probabilidades
            plt.figure(figsize=(10,6))
            plt.hist(probs[y_test==0].numpy(), bins=30, alpha=0.5, label='Classe 0 (Queda)')
            plt.hist(probs[y_test==1].numpy(), bins=30, alpha=0.5, label='Classe 1 (Alta)')
            plt.title('Distribuição das Probabilidades Previstas')
            plt.xlabel('Probabilidade Prevista')
            plt.ylabel('Frequência')
            plt.legend()
            plt.savefig(f"{self.config['output_dir']}/prob_distribution.png")
            plt.close()

if __name__ == "__main__":
    analyzer = QuantumFinanceAnalyzer(fin_config)
    analyzer.fetch_and_preprocess()
    analyzer.train_model()

