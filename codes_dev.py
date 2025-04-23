
#### quantum_sparse_nn.py > #######
import pennylane as qml
import torch
import torch.nn as nn

# -- 1) Select your backend (Qiskit Aer or local default.qubit)
try:
    from qiskit_aer import Aer
    dev = qml.device("qiskit.aer", wires=4, backend="qasm_simulator")
    print("✅ Running on Qiskit Aer backend.")
except ImportError:
    dev = qml.device("default.qubit", wires=4)
    print("⚠️ Qiskit Aer not found. Using default.qubit simulator.")

# -- 2) Define a QNode that expects exactly one data sample [4]
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs):
    """inputs is shape [4]. Condition is safe with .item() for each element."""
    for i in range(len(inputs)):
        # Convert tensor to a Python float
        if inputs[i].item() > 0.1:
            qml.RY(inputs[i], wires=i)
            qml.CNOT(wires=[i, (i+1) % len(inputs)])
    return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]

# -- 3) PyTorch module that loops over batch dimension manually
class HybridQuantumSparseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_layer = nn.Linear(4, 4)
        # No trainable quantum params, so weight_shapes={}
        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_circuit, weight_shapes={}
        )
        self.classical_output = nn.Linear(4, 2)

    def forward(self, x):
        # x shape: [B, 4]
        x = torch.tanh(self.classical_layer(x))  # shape [B, 4]

        # We must feed each sample individually to the QNode
        outputs = []
        for sample in x:  # sample shape [4]
            outputs.append(self.quantum_layer(sample))

        # outputs is length B, each element shape [4]
        outputs = torch.stack(outputs)  # shape [B, 4]
        return self.classical_output(outputs)    # shape [B, 2]

# -- 4) Simple Training
model = HybridQuantumSparseNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Make random data
X_train = torch.rand(10, 4)          # 10 samples, 4 features
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


#### lora_non_quantum_sparse_nn.py > 
import torch 
import torch.nn as nn

# 1) Definindo o módulo LoRALinear para aplicar Low-Rank Adaptation
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Peso principal que é treinado normalmente
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Matrizes de adaptação de baixa ordem (low-rank)
        self.A = nn.Parameter(torch.randn(out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, in_features))
    
    def forward(self, x):
        # Peso efetivo é a soma do peso principal com o produto A @ B
        effective_weight = self.weight + self.A @ self.B
        return torch.nn.functional.linear(x, effective_weight, self.bias)

# 2) Definindo o modelo clássico com LoRA e sparsificação inteligente
class NonQuantumLoRASparseNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Camada clássica inicial usando LoRALinear (4 -> 4)
        self.classical_layer1 = LoRALinear(4, 4, rank=2)
        self.activation = nn.Tanh()
        # Camada de sparsificação (linear simples)
        self.sparsification_layer = nn.Linear(4, 4)
        # Camada de saída que reduz de 4 para 2
        self.classical_output = nn.Linear(4, 2)
    
    def forward(self, x):
        # x tem shape [B, 4], onde B é o número de amostras
        x = self.activation(self.classical_layer1(x))  # Transforma de [B, 4] para [B, 4]
        
        # Sparsificação inteligente: zera valores cujo módulo seja menor ou igual a 0.1
        x = torch.where(x.abs() > 0.1, x, torch.zeros_like(x))
        
        # Processamento adicional na camada de sparsificação
        x = self.activation(self.sparsification_layer(x))  # Mantém a shape [B, 4]
        
        # Camada final que gera a saída (2 valores por amostra)
        return self.classical_output(x)

# 3) Treinamento do modelo clássico com LoRA e sparsificação
model = NonQuantumLoRASparseNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Gerando dados aleatórios para treinamento:
# X_train: 10 amostras, cada uma com 4 características
# y_train: 10 amostras, cada uma com 2 valores (rótulos binários)
X_train = torch.rand(10, 4)
y_train = torch.randint(0, 2, (10, 2)).float()

# Loop de treinamento
for epoch in range(50):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("✅ Training Complete!")

####  lora_quantum_sparse_nn.py > 
import pennylane as qml
import torch
import torch.nn as nn

# -------------------------
# 1) Implementação do LoRA em uma camada Linear
# -------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Peso principal (pode ser treinado normalmente)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Matrizes de adaptação de baixa ordem (low-rank)
        self.A = nn.Parameter(torch.randn(out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, in_features))
    
    def forward(self, x):
        # O peso efetivo é a soma do peso principal com o produto A x B
        effective_weight = self.weight + self.A @ self.B
        return torch.nn.functional.linear(x, effective_weight, self.bias)

# -------------------------
# 2) Seleção do Backend Quântico
# -------------------------
try:
    from qiskit_aer import Aer
    dev = qml.device("qiskit.aer", wires=4, backend="qasm_simulator")
    print("✅ Running on Qiskit Aer backend.")
except ImportError:
    dev = qml.device("default.qubit", wires=4)
    print("⚠️ Qiskit Aer not found. Using default.qubit simulator.")

# -------------------------
# 3) Definição do QNode (Circuito Quântico) com Sparsificação Interna
# -------------------------
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs):
    """
    inputs: tensor com 4 valores.
    Para cada elemento, se o valor for maior que 0.1, 
    realiza a operação quântica correspondente.
    """
    for i in range(len(inputs)):
        # Usa .item() para converter o tensor para float e evitar ambiguidades
        if inputs[i].item() > 0.1:
            qml.RY(inputs[i], wires=i)
            qml.CNOT(wires=[i, (i+1) % len(inputs)])
    return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]

# -------------------------
# 4) Definição da Rede Neural Híbrida com LoRA e Sparsificação Inteligente
# -------------------------
class HybridQuantumSparseNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Utiliza LoRA na camada clássica inicial (4 -> 4)
        self.classical_layer = LoRALinear(4, 4, rank=2)
        self.activation = nn.Tanh()
        # Camada quântica sem parâmetros treináveis
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={})
        # Camada de saída que reduz de 4 para 2
        self.classical_output = nn.Linear(4, 2)
    
    def forward(self, x):
        # x tem shape [B, 4] onde B é o tamanho do batch
        x = self.activation(self.classical_layer(x))  # Saída: [B, 4]
        
        # Aplicando Sparsificação Inteligente:
        # Zera os valores cujo módulo seja menor ou igual a 0.1.
        x = torch.where(x.abs() > 0.1, x, torch.zeros_like(x))
        
        # Processamento Individual: cada amostra é enviada separadamente para o QNode
        outputs = []
        for sample in x:  # sample tem shape [4]
            outputs.append(self.quantum_layer(sample))
        outputs = torch.stack(outputs)  # Forma um tensor com shape [B, 4]
        
        return self.classical_output(outputs)  # Saída final: [B, 2]

# -------------------------
# 5) Treinamento Simples
# -------------------------
model = HybridQuantumSparseNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Dados aleatórios: 10 amostras, 4 features cada; rótulos com 2 valores
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

####  mpt_gpu_quantum_sparse_nn.py >
import pennylane as qml
import torch
import torch.nn as nn

# -------------------------
# 1) Seleção do Backend Quântico
# -------------------------
try:
    from qiskit_aer import Aer
    dev = qml.device("qiskit.aer", wires=4, backend="qasm_simulator")
    print("✅ Running on Qiskit Aer backend.")
except ImportError:
    dev = qml.device("default.qubit", wires=4)
    print("⚠️ Qiskit Aer not found. Using default.qubit simulator.")

# -------------------------
# 2) Definição do QNode (Circuito Quântico)
# -------------------------
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs):
    """
    inputs: tensor com 4 valores.
    Para cada elemento, se o valor for maior que 0.1,
    realiza RY e uma porta CNOT entre qubits.
    """
    for i in range(len(inputs)):
        if inputs[i].item() > 0.1:
            qml.RY(inputs[i], wires=i)
            qml.CNOT(wires=[i, (i+1) % len(inputs)])
    return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]

# -------------------------
# 3) Definição da Rede Neural Híbrida
# -------------------------
class HybridQuantumSparseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_layer = nn.Linear(4, 4)
        self.activation = nn.Tanh()
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={})
        self.classical_output = nn.Linear(4, 2)

    def forward(self, x):
        # x tem shape [B, 4]
        x = self.activation(self.classical_layer(x))  # [B, 4]
        # Enviar cada amostra individualmente para o QNode
        outputs = []
        for sample in x:  # sample shape: [4]
            outputs.append(self.quantum_layer(sample))
        outputs = torch.stack(outputs)  # [B, 4]
        return self.classical_output(outputs)  # [B, 2]

# -------------------------
# 4) Configuração do Treinamento com Mixed Precision
# -------------------------
model = HybridQuantumSparseNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Verifica se há GPU disponível para usar mixed precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_train = torch.rand(10, 4, device=device)
y_train = torch.randint(0, 2, (10, 2), device=device).float()

# Criar um GradScaler para lidar com a escala da perda
scaler = torch.cuda.amp.GradScaler()

# -------------------------
# 5) Loop de Treinamento com Mixed Precision Training
# -------------------------
for epoch in range(50):
    optimizer.zero_grad()
    # Usa autocast para realizar cálculos em float16 quando possível
    with torch.cuda.amp.autocast():
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
    # Escala a perda, faz o backward e atualiza os parâmetros
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("✅ Training Complete!")

####  mpt_gpu&cpu_quantum_sparse_nn copy.py >
import pennylane as qml
import torch
import torch.nn as nn

# -------------------------
# 1) Seleção do Backend Quântico
# -------------------------
try:
    from qiskit_aer import Aer
    dev = qml.device("qiskit.aer", wires=4, backend="qasm_simulator")
    print("✅ Running on Qiskit Aer backend.")
except ImportError:
    dev = qml.device("default.qubit", wires=4)
    print("⚠️ Qiskit Aer not found. Using default.qubit simulator.")

# -------------------------
# 2) Definição do QNode (Circuito Quântico)
# -------------------------
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs):
    """
    inputs: tensor com 4 valores.
    Para cada elemento, se o valor for maior que 0.1, 
    realiza RY e uma porta CNOT.
    """
    for i in range(len(inputs)):
        if inputs[i].item() > 0.1:
            qml.RY(inputs[i], wires=i)
            qml.CNOT(wires=[i, (i + 1) % len(inputs)])
    return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]

# -------------------------
# 3) Definição da Rede Neural Híbrida
# -------------------------
class HybridQuantumSparseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_layer = nn.Linear(4, 4)
        self.activation = nn.Tanh()
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={})
        self.classical_output = nn.Linear(4, 2)

    def forward(self, x):
        # x shape: [B, 4]
        x = self.activation(self.classical_layer(x))  # shape [B, 4]
        outputs = []
        # Processamento individual: cada amostra é enviada ao QNode
        for sample in x:  # sample shape: [4]
            outputs.append(self.quantum_layer(sample))
        outputs = torch.stack(outputs)  # shape [B, 4]
        return self.classical_output(outputs)  # shape [B, 2]

# -------------------------
# 4) Configuração do Treinamento com Mixed Precision Training
# -------------------------
model = HybridQuantumSparseNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Seleciona o dispositivo: GPU (CUDA) se disponível, caso contrário CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_train = torch.rand(10, 4, device=device)
y_train = torch.randint(0, 2, (10, 2), device=device).float()

# Configura o GradScaler somente se CUDA estiver disponível
if torch.cuda.is_available():
    scaler = torch.amp.GradScaler(device="cuda")
else:
    scaler = None

# -------------------------
# 5) Loop de Treinamento
# -------------------------
for epoch in range(50):
    optimizer.zero_grad()
    # Se CUDA estiver disponível, usa autocast para mixed precision
    if torch.cuda.is_available() and scaler is not None:
        with torch.amp.autocast(device_type="cuda"):
            preds = model(X_train)
            loss = loss_fn(preds, y_train)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("✅ Training Complete!")

####  non_quantum_sparse_nn.py >
import torch
import torch.nn as nn

# Definindo o modelo clássico sem camada quântica
class NonQuantumSparseNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Camada clássica inicial: transforma o vetor de 4 elementos em outro vetor de 4 elementos
        self.classical_layer1 = nn.Linear(4, 4)
        # Função de ativação Tanh
        self.activation = nn.Tanh()
        # Camada que simula a "sparsificação": aplica uma transformação linear adicional
        self.sparsification_layer = nn.Linear(4, 4)
        # Camada de saída que reduz de 4 para 2 elementos
        self.classical_output = nn.Linear(4, 2)

    def forward(self, x):
        # x tem shape [B, 4] (B = número de amostras do batch)
        x = self.activation(self.classical_layer1(x))  # shape [B, 4]
        # Simula a sparsificação: para cada valor, zera os que forem menores que 0.1
        x = torch.where(x > 0.1, x, torch.zeros_like(x))
        # Processamento adicional na camada de "sparsificação"
        x = self.activation(self.sparsification_layer(x))  # shape [B, 4]
        # Camada final que gera a saída (2 valores por amostra)
        return self.classical_output(x)

# Treinamento do modelo clássico
model = NonQuantumSparseNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Gerando dados aleatórios para treinamento:
# X_train: 10 amostras, cada uma com 4 características
# y_train: 10 amostras, cada uma com 2 valores (rótulos binários)
X_train = torch.rand(10, 4)
y_train = torch.randint(0, 2, (10, 2)).float()

# Loop de treinamento
for epoch in range(50):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("✅ Training Complete!")

####  q8_non_quantum_sparse_nn.py > 
import torch
import torch.nn as nn
import torch.quantization as quant

# 1) Definindo o módulo LoRALinear para aplicar Low-Rank Adaptation
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Peso principal treinável
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Matrizes de adaptação de baixa ordem (low-rank)
        self.A = nn.Parameter(torch.randn(out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, in_features))
    
    def forward(self, x):
        # O peso efetivo é a soma do peso principal com o produto A @ B
        effective_weight = self.weight + self.A @ self.B
        return torch.nn.functional.linear(x, effective_weight, self.bias)

# 2) Definindo o modelo clássico com LoRA e sparsificação inteligente
class NonQuantumLoRASparseNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Camada clássica inicial usando LoRALinear (4 -> 4)
        self.classical_layer1 = LoRALinear(4, 4, rank=2)
        self.activation = nn.Tanh()
        # Camada de sparsificação (linear simples)
        self.sparsification_layer = nn.Linear(4, 4)
        # Camada de saída que reduz de 4 para 2
        self.classical_output = nn.Linear(4, 2)
    
    def forward(self, x):
        # x tem shape [B, 4], onde B é o número de amostras
        x = self.activation(self.classical_layer1(x))  # Transforma de [B, 4] para [B, 4]
        # Sparsificação inteligente: zera valores cujo módulo seja menor ou igual a 0.1
        x = torch.where(x.abs() > 0.1, x, torch.zeros_like(x))
        # Processamento adicional na camada de sparsificação
        x = self.activation(self.sparsification_layer(x))  # Mantém a shape [B, 4]
        # Camada final que gera a saída (2 valores por amostra)
        return self.classical_output(x)

# 3) Treinamento do modelo clássico com LoRA e sparsificação
model = NonQuantumLoRASparseNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Gerando dados aleatórios para treinamento:
# X_train: 10 amostras, cada uma com 4 características
# y_train: 10 amostras, cada uma com 2 valores (rótulos binários)
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

# 4) Aplicando Quantização Dinâmica (Q8) às camadas lineares
# Incluímos nn.Linear e nossa classe LoRALinear
model_quant = quant.quantize_dynamic(model, {nn.Linear, LoRALinear}, dtype=torch.qint8)
print("✅ Quantization Applied!")

# 5) Teste de inferência com o modelo quantizado
with torch.no_grad():
    preds_quant = model_quant(X_train)
    loss_quant = loss_fn(preds_quant, y_train)
    print(f"Quantized Model Loss: {loss_quant.item():.4f}")

q8_quantum_sparse_nn.py >

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


####  q8+_quantum_sparse_nn.py > 

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
