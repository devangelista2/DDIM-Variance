from diffusers import (
    DDIMPipeline,
    UNet2DModel,
    DDIMScheduler,
)
import torch
import numpy as np
from miscellaneous import utilities


def generate_random(
    cfg,
    n_samples=4,
    seed=None,
):
    # Set device
    device = utilities.set_device()

    # Initialize model and schedule
    model = UNet2DModel.from_pretrained(cfg["path"]["model"])
    scheduler = DDIMScheduler.from_pretrained(cfg["path"]["scheduler"])

    # Initialize the pipeline with the model and scheduler
    pipeline = DDIMPipeline(unet=model, scheduler=scheduler).to(device)

    # Generate images
    if seed is not None:
        seed = torch.Generator().manual_seed(seed)

    x_gen_list = pipeline(
        batch_size=n_samples,
        num_inference_steps=cfg["inversion"]["ODE"]["n_timesteps_generation"],
        generator=seed,
    ).images

    # Convert list of images to tensor
    for i in range(len(x_gen_list)):
        # Read and convert
        x_tmp = utilities.normalize(
            torch.tensor(
                np.array(
                    x_gen_list[i],
                    dtype=np.float32,
                ),
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Concatenate
        if i == 0:
            x_gen = x_tmp
        else:
            x_gen = torch.cat((x_gen, x_tmp), dim=0)

    return x_gen


def generate_from_z(
    cfg,
    z,
    n_timesteps,
):
    # Set device
    device = utilities.set_device()

    # Initialize model and schedule
    model = UNet2DModel.from_pretrained(cfg["path"]["model"]).to(device)
    scheduler = DDIMScheduler()

    # Set model parameter requires_grad to False to reduce memory consumption
    for param in model.parameters():
        param.requires_grad = False

    # Set schedule timesteps
    scheduler.set_timesteps(n_timesteps)

    for t in reversed(range(n_timesteps)):
        with torch.no_grad():
            predicted_noise = model(z, t).sample
            z = scheduler.step(predicted_noise, t, z).prev_sample

    return z
