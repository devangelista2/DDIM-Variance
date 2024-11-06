from miscellaneous import utilities
from functionalities import generate
import time

import os
import torch

# Select config experiments
cfg = utilities.load_config("experiments/DDIM_inversion.json")
image_path = os.path.join("../data/Mayo/test/C081/10.png")

########## SETTING THINGS UP #########
device = utilities.set_device()
print(f"Device used: {device}.")

# Setup direct inversion task
start_time = time.time()
x_rec = generate.generate_random(cfg, n_samples=16)
time_end = time.time()

print(f"Done (in {time.time() - start_time:0.4f}s)")

# Get variance
print(torch.mean(torch.var(x_rec, dim=0)))
