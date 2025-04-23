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


# SECOND 

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

OUTPUT > 
✅ Running on Qiskit Aer backend.
Epoch 0 | Loss: 0.2544
Epoch 10 | Loss: 0.2296
Epoch 20 | Loss: 0.2224
Epoch 30 | Loss: 0.2167
Epoch 40 | Loss: 0.2043
✅ Training Complete!

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

---

### 1) Quantum Backend Selection

- **First script (`q8+_quantum_sparse_nn.py`):**  
  - Tries **three** backends in order:
    1. **`lightning.gpu`** via `pennylane_lightning_gpu` (with `diff_method="adjoint"`)  
    2. **`qiskit.aer`** if Lightning isn’t installed  
    3. **`default.qubit`** as a final fallback  
  - Prints a confirmation of whichever backend is chosen.

- **Second script:**  
  - Only attempts **`qiskit.aer`**, then falls back to `default.qubit`.  
  - No GPU-specific PennyLane Lightning plugin, and no explicit `diff_method`.

---

### 2) QNode Definition and Differentiability

| Aspect                     | First Script                                                              | Second Script                                   |
|----------------------------|---------------------------------------------------------------------------|-------------------------------------------------|
| **Decorator**              | `@qml.qnode(dev, interface="torch", diff_method="parameter-shift")`       | `@qml.qnode(dev, interface="torch")`            |
| **Threshold check inside** | Uses **tensor comparison** `if inputs[i] > 0.1` (keeps it differentiable)  | Uses **`.item()`** `if inputs[i].item() > 0.1`  |
| **Impact on gradients**    | All operations remain on the computational graph (masking is backpropable) | Calling `.item()` detaches and blocks gradients |

---

### 3) Sparsity Mechanism

- **First script (Advanced):**  
  - After a classical `Linear → Tanh` layer, applies a **learnable soft-threshold**:  
    ```python
    mask = torch.sigmoid(self.alpha * (x.abs() - 0.1))
    x = x * mask
    ```  
  - Here `self.alpha` (initialized to 10.0) controls the “hardness” of the threshold, allowing the model to learn how sparse to make its quantum inputs.

- **Second script (Basic):**  
  - **No soft-threshold** or masking: every activation flows directly into the quantum circuit.

---

### 4) Model Class Structure

| Feature                 | AdvancedQuantumSparseNN                                    | HybridQuantumSparseNN                    |
|-------------------------|-------------------------------------------------------------|-------------------------------------------|
| **Learnable α-parameter**     | ✓                                                           | ✗                                         |
| **Soft-threshold mask**       | ✓                                                           | ✗                                         |
| **`classical_layer`→`activation`→mask** | ✓                                                | `classical_layer`→`activation` only       |
| **Quantum layer**        | `qml.qnn.TorchLayer(quantum_circuit, weight_shapes={})`      | Same                                      |
| **Classical output**      | Linear(4→2)                                                 | Linear(4→2)                               |

---

### 5) Device Placement for Training

- **First script:**  
  - Runs everything on CPU by default (no `.to(device)`).

- **Second script:**  
  - Detects CUDA (`torch.cuda.is_available()`) and moves both model and data to GPU if possible:
    ```python
    device = torch.device("cuda" if … else "cpu")
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    ```

---

### 6) Quantization Module Import

- **First script:**  
  ```python
  import torch.ao.quantization as quant
  ```
  (newer alias in the PyTorch `ao` / Accelerated Optimizer namespace)

- **Second script:**  
  ```python
  import torch.quantization as quant
  ```
  (classic quantization API)

Both then apply **dynamic quantization** to the two classical `Linear` layers:

```python
model.classical_layer = quant.quantize_dynamic(model.classical_layer, {nn.Linear}, dtype=torch.qint8)
model.classical_output = quant.quantize_dynamic(model.classical_output, {nn.Linear}, dtype=torch.qint8)
```

---

### 7) Summary of Functional Impact

- The **Advanced** version gives you **learnable sparsity**, keeping the quantum circuit “light” by gating out small activations in a fully differentiable way, and optimizes for GPU-backend speed with PennyLane Lightning.
- The **Basic** version is a simpler hybrid model without sparsity control, using a straightforward `.item()` gating (which breaks gradient flow) and only Qiskit Aer / default.qubit backends.

---

By choosing the **advanced** pattern, you gain fine-grained, gradient-driven control over how much information is fed into the costly quantum circuit—trading a bit more implementation complexity for potentially much greater efficiency and trainability.

True Differentiable Sparsity

Soft-threshold mask: It uses

python
Copy
Edit
mask = sigmoid(α · (|x| – 0.1))
x = x * mask
with α as a learnable parameter. This keeps the entire sparsification step on the computation graph, so the network can adapt how aggressively it prunes small activations during training.

Basic version lacks any masking, so every activation—even noise—feeds into the quantum circuit, wasting precious quantum resources.

Gradient Integrity in the QNode

The advanced script’s if inputs[i] > 0.1 check uses the tensor directly, preserving upstream gradients.

In the basic script, inputs[i].item() > 0.1 detaches the tensor and breaks gradient flow through that conditional, which hampers end-to-end learning.

Backend Optimization

The advanced version automatically falls back through three backends—lightning.gpu (with the adjoint method), then Qiskit Aer, then default.qubit—so you seamlessly exploit GPU acceleration or the fastest simulator available.

The basic version only tries Qiskit Aer and default.qubit, missing out on PennyLane Lightning’s adjoint-mode speedups.

Modern Quantization API

By importing from torch.ao.quantization, the advanced script is aligned with PyTorch’s evolving “Accelerated Optimization” namespace; this is forward-compatible as PyTorch’s quantization APIs mature.

Both apply the same dynamic quantization to the classical layers, but the advanced approach is already using the newer module structure.

Maintainability & Extensibility

The advanced class is explicitly parameterized (self.alpha), easier to monitor and adjust.

Its structure cleanly separates classical encoding, sparsification, quantum evaluation, and classical decoding—ideal for iterative research on threshold strategies or more complex quantum circuits.

When the basic script might suffice

Rapid prototyping or educational demos, where you just want a “quantum toy” without worrying about gradient fidelity or run-time efficiency.

Resource-constrained environments where adding the PennyLane Lightning plugin or tuning a learnable threshold isn’t yet justified.
