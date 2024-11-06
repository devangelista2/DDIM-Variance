import math
import time

import matplotlib.pyplot as plt
import torch
from diffusers import DDIMScheduler

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

# Set schedule timesteps
scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
)

alpha_t = scheduler.alphas_cumprod
g_t = torch.zeros_like(alpha_t)
frac_t = torch.zeros_like(alpha_t)

T = len(alpha_t)
for t in range(1, len(alpha_t)):
    g_t[t] = (
        math.sqrt(1 - alpha_t[T - t - 1])
        - math.sqrt(alpha_t[T - t - 1] / alpha_t[T - t] * (1 - alpha_t[T - t]))
    ) ** 2

    frac_t[t] = g_t[t] / alpha_t[T - t - 1]

cum_frac_t = torch.cumsum(frac_t, dim=0)
pred_variances = torch.zeros_like(cum_frac_t)
for i in range(len(cum_frac_t)):
    pred_variances[i] = cum_frac_t[-i - 1]
scheduler.set_timesteps(1000)

# Initialize reverse loop
torch.manual_seed(42)
x_t = torch.randn((5, 1, 128, 128), device=device)
real_var = torch.zeros((T,))
for i, t in enumerate(scheduler.timesteps):
    with torch.no_grad():
        eps_t = eps(x_t, t)
        x_t = scheduler.step(eps_t, t, x_t).prev_sample

        print(
            f"t = {t.item()}, Var(eps_theta(t)) = {torch.mean(torch.var(eps_t, dim=0)).item():0.4f},",
            f"Var(x_t) = {torch.mean(torch.var(x_t, dim=0)).item():0.4f},",
        )
        real_var[i] = torch.mean(torch.var(x_t, dim=0))


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


plt.plot(normalize(pred_variances))
plt.plot(normalize(real_var))
plt.grid()
plt.show()
