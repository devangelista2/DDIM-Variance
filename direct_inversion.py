from miscellaneous import utilities
from functionalities import generate, inversion
import time

import os

# Select config experiments
cfg = utilities.load_config("experiments/DDIM_inversion.json")
image_path = os.path.join("../data/Mayo/test/C081/10.png")

########## SETTING THINGS UP #########
device = utilities.set_device()
print(f"Device used: {device}.")

x_true = utilities.get_image_from_path(image_path).to(device)

# Setup direct inversion task
start_time = time.time()
print(
    f"Inverting image with {cfg["inversion"]["n_timesteps_inversion"]}",
    f"reverse steps.",
)

z = inversion.ODE_inversion(
    cfg,
    x_true,
    n_timesteps=cfg["inversion"]["ODE"]["n_timesteps_inversion"],
)
x_rec = generate.generate_from_z(
    cfg,
    z,
    n_timesteps=cfg["inversion"]["ODE"]["n_timesteps_generation"],
)
time_end = time.time()
print(f"Done (in {time.time() - start_time:0.4f}s)")

# Visualization
utilities.show_image_list(
    image_list=[x_rec, z, x_true], titles=["x_rec", "z", "x_true"]
)
