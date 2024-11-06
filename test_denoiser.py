import time

import matplotlib.pyplot as plt
import torch

from functionalities import generate
from miscellaneous import utilities

# Select config experiments
BASE_PATH = "model_weights/DDIM_128"
n_timesteps = 1000

########## SETTING THINGS UP #########
device = utilities.set_device()
print(f"Device used: {device}.")

# Initialize denoiser and generator
G = generate.Generator(base_path=BASE_PATH)
eps = G.eps

# Setup generation task
avg_over_t = torch.zeros((n_timesteps))
over_t_avg = torch.zeros((n_timesteps))
over_0_avg = torch.zeros((n_timesteps))
start_time = time.time()
z = torch.randn((5, 1, 128, 128), device=device)
for t in range(n_timesteps):
    if t % 100 == 0:
        print(
            f"Step ({t+1} / {n_timesteps}) done (in {time.time() - start_time:0.2f}s)."
        )
    eps_t = eps(z, t).cpu()
    eps_E_t = eps(z.mean(dim=0).unsqueeze(0), t).cpu()
    eps_0_t = eps(torch.zeros((5, 1, 128, 128), device=device), t).cpu()

    avg_over_t[t] = eps_t.mean()
    over_t_avg[t] = eps_E_t.mean()
    over_0_avg[t] = eps_0_t.mean()

# Visualize
plt.plot(avg_over_t)
plt.plot(over_t_avg)
plt.plot(over_0_avg)
plt.grid()
plt.xlabel(r"$t$")
plt.legend(
    [
        r"$\mathbb{E}[\epsilon_\theta(N(0, I), t)]$",
        r"$\epsilon_\theta(\mathbb{E}[N(0, I)], t)$",
        r"$\epsilon_\theta(0, t)$",
    ]
)
plt.show()
