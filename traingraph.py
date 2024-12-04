import torch
import torch.utils.tensorboard as tb
import numpy as np
import matplotlib.pyplot as plt
import time

epochs = []
losses = []
times = []

with open('training_losses.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        epoch = int(parts[0].split()[-1])
        loss = float(parts[1].split()[-1])
        time_str = parts[2].split('=')[-1].strip()
        time = float(time_str.replace('sec', '').strip())
        epochs.append(epoch)
        losses.append(loss)
        times.append(time)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.plot(epochs, losses, 'b-', label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Over Time')
ax1.grid(True)
ax1.legend()

ax2.plot(epochs, times, 'r-', label='Epoch Duration')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Training Time per Epoch')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()