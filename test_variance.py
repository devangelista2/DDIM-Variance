import time

import matplotlib.pyplot as plt
import torch

from functionalities import generate
from miscellaneous import utilities

# Select config experiments
BASE_PATH = "model_weights/DDIM_128"

########## SETTING THINGS UP #########
device = utilities.set_device()
print(f"Device used: {device}.")

# Initialize denoiser and generator
G = generate.Generator(base_path=BASE_PATH)
eps = G.eps

# Setup generation task
start_time = time.time()
x_T = torch.randn((1, 1, 128, 128), device=device)
x_0 = G(x_T, n_timesteps=30, t_tilde=0).cpu()
time_end = time.time()

print(f"Done (in {time.time() - start_time:0.4f}s)")

# Visualize
plt.imshow(x_0[0, 0], cmap="gray")
plt.axis("off")
plt.show()

# Get variance
# print(torch.mean(torch.var(x_0, dim=0)))
