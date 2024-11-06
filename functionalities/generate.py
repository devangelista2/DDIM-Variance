import numpy as np
import torch
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel

from miscellaneous import utilities


class Denoiser:
    def __init__(self, base_path, *args, **kwargs):
        # Set device
        self.device = utilities.set_device()

        # Initialize model and schedule
        self.model = UNet2DModel.from_pretrained(base_path + "/unet").to(self.device)

        # Set model parameter requires_grad to False to reduce memory consumption
        for param in self.model.parameters():
            param.requires_grad = False

    def __call__(self, x_t, t, *args, **kwargs):
        with torch.no_grad():
            predicted_noise = self.model(x_t, t).sample
        return predicted_noise


class Generator:
    def __init__(self, base_path, *args, **kwargs):
        # Set device
        self.device = utilities.set_device()

        # Initialize scheduler and Denoiser
        self.eps = Denoiser(base_path=base_path)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )

    def __call__(self, x_T, n_timesteps=100, t_tilde=0, *args, **kwargs):
        # Set schedule timesteps
        self.scheduler.set_timesteps(n_timesteps)

        # Initialize reverse loop
        x_t = x_T
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                eps_t = self.eps(x_t, t)
                x_t = self.scheduler.step(eps_t, t, x_t).prev_sample

        return x_t
