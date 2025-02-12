import torch
import matplotlib.pyplot as plt
from torch.nn import GELU
import torch.nn as nn

x = torch.linspace(-3, 3, 100)
# Convert tensor to list manually
x_list = [float(val) for val in x]

gelu, relu = GELU(), nn.ReLU()
y_gelu, y_relu = gelu(x), relu(x)

# Convert y values to lists too
y_gelu_list = [float(val) for val in y_gelu]
y_relu_list = [float(val) for val in y_relu]

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu_list, y_relu_list], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x_list, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()