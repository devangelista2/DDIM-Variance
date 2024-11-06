from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import math

preprocess = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)

    return cfg


def set_device():
    if torch.cuda.is_available():
        return "cuda"

    elif torch.mps.is_available():
        return "mps"

    else:
        return "cpu"


def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def get_image_from_path(path):
    x = Image.open(path).convert("L")
    x_np = np.array(preprocess(x), dtype=np.float32)
    x_torch = torch.tensor(x_np).unsqueeze(0)
    return x_torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().squeeze().cpu().numpy()


def save_metrics_to_file(filename: str, metrics: list, header: list[str]) -> None:
    """
    Save metrics to a file with a specified header.

    Args:
        filename (str): The name of the file where metrics will be saved.
        metrics (list): A list of metrics, where each metric is a list of values.
                        The first element of each metric list is a string identifier.
        header (list[str]): A list of strings representing the header for the metrics file.

    Returns:
        None
    """
    with open(filename, "w") as f:
        f.write(" : ".join(header) + "\n")
        for metric in metrics:
            if metric[0] != "LINE":
                f.write(metric[0] + " : ")
                for value in metric[1:]:
                    f.write(f"{value:.4f} ")
                f.write("\n")


def load_metrics_from_file(filename: str) -> list:
    """
    Load metrics from a file.

    The file is expected to have a header line followed by lines of metric data.
    Each line of metric data should have a name followed by values separated by colons.
    The function skips lines where the name is "LINE".

    Args:
        filename (str): The path to the file containing the metrics.

    Returns:
        list: A list of tuples where the first element is the metric name (str) and
              the subsequent elements are the metric values (float or str).
    """
    metrics = []
    with open(filename, "r") as f:
        header = f.readline().strip().split(" : ")
        num_values = len(header) - 1
        for line in f:
            name = line.split(":")[0].strip()
            if name == "LINE":
                continue
            values = line.split(":")[1].strip().split()
            values = [
                float(value) if i < num_values else value
                for i, value in enumerate(values)
            ]
            metrics.append((name, *values))
    return metrics


def show_image(image_tensor: torch.Tensor, title: str = ""):
    image_np = tensor_to_numpy(image_tensor)
    plt.imshow(image_np, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_image_list(image_list, titles):
    n_images = len(image_list)

    figure = plt.figure(figsize=(5 * n_images, 5))
    for i in range(n_images):
        figure.add_subplot(1, n_images, i + 1)
        plt.imshow(normalize(tensor_to_numpy(image_list[i])), cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.show()
