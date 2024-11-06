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
# Setup generation task
variances = torch.zeros((100))
start_time = time.time()
x_T = torch.randn((5, 1, 128, 128), device=device)
for i, n_timesteps in enumerate(range(2, 300, 10)):
    x_0 = G(x_T, n_timesteps=n_timesteps).cpu()
    variances[i] = torch.mean(torch.var(x_0, dim=0))

    print(
        f"Step ({i+1}) -> ({n_timesteps}) done (in {time.time() - start_time:0.2f}s)."
    )
variances = variances[: i + 1]

# Visualize
plt.plot(torch.arange(2, 300, 10), variances)
plt.grid()
plt.xlabel(r"n_timesteps")
plt.legend(
    [
        r"$\mathbb{E}[Var(G(N(0, I), t))]$",
    ]
)
plt.show()
