# %%
from accelerate import Accelerator
from diffusers import DDIMPipeline
from diffusers import UNet2DModel
from diffusers import DDIMScheduler
import torch
import PIL.Image
import numpy as np

# %%
#Configurazioni

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128                     # the generated image resolution
    train_batch_size = 4
    eval_batch_size = 4                  # how many images to sample during evaluation
    num_epochs = 300
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "fp16"              # `no` for float32, `fp16` for automatic mixed precision
    dataset_name = "../../datasets/Mayos"
    output_dir = "outputs/output_DDIM_128"
    model_path = "outputs/output_DDIM_128/unet"
    scheduler_path = "outputs/output_DDIM128/scheduler"
    train_new_model = True
    train_new_scheduler = True
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

# %%
#Data Loading

from datasets import load_dataset

dataset = load_dataset(config.dataset_name, split="train")
#dataset = dataset.filter(lambda x: x["label"] == 9)
print("Dataset len: "+str(dataset.shape[0]))

# %%
#Preprocessing Immagini
from torchvision import transforms
import matplotlib.pyplot as plt

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# Get the first image
first_image = dataset[0]["image"]

# Display the first image before transformation
plt.figure(figsize=(10, 5))

# Convert the PIL image to numpy array for displaying
plt.subplot(1, 2, 1)
plt.imshow(first_image,cmap='gray')
plt.title("Original Image")

def transform(examples):
    images = [preprocess(image) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

# Display the transformed image
plt.subplot(1, 2, 2)
# Get the first image
transform_image = dataset[0]["images"]
plt.imshow(transform_image.numpy().squeeze(0),cmap='gray')
plt.title("Transformed Image")


train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# %%
if config.train_new_model:
    model = UNet2DModel(
        sample_size=config.image_size,        # the target image resolution
        in_channels=1,          # the number of input channels, 3 for RGB images
        out_channels=1,         # the number of output channels
        layers_per_block=2,                                 # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 128, 256),      # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",      # a regular ResNet downsampling block
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",        # a regular ResNet upsampling block
            "AttnUpBlock2D",    # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    print("New model created")
else:
    model = UNet2DModel.from_pretrained(config.model_path)
    print("Model loaded from", config.model_path)

sample_image = dataset[0]["images"].unsqueeze(0)
print("Input shape:", sample_image.shape)

print("Output shape:", model(sample_image, timestep=0).sample.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters())

print(f"Total number of parameters: {total_params}")

# %%
#Evaluation
from diffusers.utils import make_image_grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=int(config.eval_batch_size/2), cols=int(config.eval_batch_size/2))

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

# %%
#Accellerated Training Setup
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
import os

from diffusers.optimization import get_cosine_schedule_with_warmup

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(torch.float32)
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)

# %%
#Noising and SetUp

if config.train_new_scheduler:
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    print("New Scheduler Loaded")
else:
    noise_scheduler = DDIMScheduler.from_pretrained(config.scheduler_path)
    print("Scheduler Loaded from "+config.scheduler_path)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

# %%
#Training

train_loop(config,model,noise_scheduler,optimizer,train_dataloader,lr_scheduler)


