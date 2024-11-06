from diffusers import (
    UNet2DModel,
    DDIMInverseScheduler,
)
import torch
import numpy as np
from miscellaneous import utilities
from functionalities import generate


def ODE_inversion(
    cfg,
    x,
    n_timesteps,
):
    # Set device
    device = utilities.set_device()

    # Initialize model and schedule
    model = UNet2DModel.from_pretrained(cfg["path"]["model"]).to(device)
    scheduler = DDIMInverseScheduler()

    # Set schedule timesteps
    scheduler.set_timesteps(n_timesteps)

    for t in range(n_timesteps):
        with torch.no_grad():
            predicted_noise = model(x, t).sample
            x = scheduler.step(predicted_noise, t, x).prev_sample

    return x


def optimizer_inversion(
    cfg,
    x_true,
    n_timesteps,
    opt_steps,
    loss_fn,
):
    """
    The function performs the following steps:
    1. Initializes the device (GPU if available, otherwise CPU).
    2. Defines a custom loss function that combines reconstruction loss and regularization.
    3. Initializes noise tensor `z` with gradients enabled.
    4. Sets up the optimizer (AdamW) for `z`.
    5. Saves initial noise and target images.
    6. Returns the final optimized noise tensor and image.
    """
    # Set device
    device = utilities.set_device()

    # Starting point
    z = torch.randn_like(x_true).to(device).requires_grad_(True)

    # Set optimizer
    optimizer = torch.optim.AdamW([z], lr=cfg["inversion"]["optimization"]["lr"])

    for i in range(opt_steps):
        # Compute x generation
        x_gen = generate.generate_from_z(cfg, z, n_timesteps)

        # Restart gradient of optimizer
        optimizer.zero_grad()
        loss = loss_fn(x_gen, x_true, z)
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

    return z
