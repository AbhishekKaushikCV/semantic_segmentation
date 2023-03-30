import cv2
import torch
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


from constants import num_classes

def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module,
                device: str,
                masks2labels: callable,
                color_segmap: callable) -> Tuple[float, float]:
    """
    Train a PyTorch model on a dataset using a specified optimizer and loss function.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (torch.utils.data.DataLoader): A PyTorch DataLoader object representing the training dataset.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to use for training.
        criterion (torch.nn.Module): The PyTorch loss function to use for training.
        device (str): The device (e.g. "cpu" or "cuda") on which to train the model.
        masks2labels (callable): A function to convert a tensor of masks to a tensor of labels.
        color_segmap (callable): A function to encode a tensor of labels into a colored segmentation map.

    Returns:
        A tuple containing the average training loss and pixel accuracy.
    """
    model.train()
    train_loss = 0.0
    train_pixel_acc = 0.0
    for idx, data in enumerate(train_loader):
        features, labels = data
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(features)
        
        out_img = masks2labels(outputs.squeeze().detach().cpu().numpy().astype('uint8'))
        out_rgb_img = color_segmap(out_img, num_classes)
        cv2.imwrite(f'./samples/outputs/output{idx}.png', out_rgb_img)
        
        labels_img = masks2labels(labels.squeeze().detach().cpu().numpy().astype('uint8'))
        labels_rgb_img = color_segmap(labels_img, num_classes)
        cv2.imwrite(f'./samples/labels/labels{idx}.png', labels_rgb_img)
        #calculate loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        predicted_labels = torch.argmax(outputs, dim=1)      
        max_labels = torch.argmax(labels, dim=1)
        train_loss += loss.item() * features.size(0)
        train_pixel_acc += torch.sum(predicted_labels == max_labels).item()

    train_loss /= len(train_loader.dataset)
    train_pixel_acc /= (len(train_loader.dataset) * labels.size(-1) * labels.size(-2))
    return train_loss, train_pixel_acc
    
# to encode masks into colored segmentation map

def color_segmap(image: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Given a segmentation map image and the number of classes,
    this function assigns colors to each class and returns the resulting RGB image.
    
    Args:
    - image (np.ndarray): a segmentation map image where each pixel value corresponds to a class label
    - num_classes (int): the number of classes in the segmentation map
    
    Returns:
    - rgb (np.ndarray): the resulting RGB image with colors assigned to each class
    """
    label_colors = np.array([(0, 0, 0),  # 0=road
                             # 1=sidewalk, 2=building, 3=wall, 4=fence, 5=pole
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=trafic light, 7=traffice sign, 8=vegetation, 9=terrain, 10=sky
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=person, 12=rider, 13=car, 14=truck, 15=bus
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=train, 17=motorcycle, 18=bicycle, 19=unlabelled
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (255, 255, 255)])
    
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, num_classes):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb


# convert masks2labels

def masks2labels(masks: np.ndarray) -> np.ndarray:
    """
    Converts an array of one-hot encoded masks into an array of integer labels.

    Args:
        masks: An array of shape (num_classes, height, width) representing one-hot encoded masks.

    Returns:
        An integer array of shape (height, width) representing the class label for each pixel.

    """
    labels = np.argmax(masks, axis=0)
    labels[labels == masks.shape[0] - 1] = -1  # map the 19 index back to unlabelled
    return labels

# plot training results

def plot_training_results(train_losses: List[float], train_pixel_accs: List[float]) -> None:
    """
    Plots the training loss and pixel accuracy on the same graph.

    Args:
        train_losses (list): A list of training losses.
        train_pixel_accs (list): A list of training pixel accuracies.

    Returns:
        None
    """
    plt.plot(train_losses, label='Training Loss')
    plt.plot(train_pixel_accs, label='Training Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Results')
    plt.legend()
    plt.show()


from PIL import Image, ImageDraw
import os

def create_gif(image_dir: str, output_dir: str, gif_name: str):
    """
    Combines the first 12 images from 'labels/' and 'outputs/' directories
    and saves them as a GIF.

    Args:
        image_dir (str): Path to the directory containing the images.
        output_dir (str): Path to the directory to save the GIF.
        gif_name (str): Name of the GIF file to be saved.

    Returns:
        None
    """
    # Get the first 12 images from the 'labels/' directory
    label_path = os.path.join(image_dir, 'labels')
    label_files = sorted(os.listdir(label_path))
    label_images = [Image.open(os.path.join(label_path, f)) for f in label_files]

    # Get the first 12 images from the 'outputs/' directory
    output_path = os.path.join(image_dir, 'outputs')
    output_files = sorted(os.listdir(output_path))
    output_images = [Image.open(os.path.join(output_path, f)) for f in output_files]

    # Combine the images horizontally
    combined_images = []
    for i in range(len(label_files)):
        combined_image = Image.new('RGB', (label_images[i].width * 2, label_images[i].height))
        combined_image.paste(label_images[i], (0, 0))
        combined_image.paste(output_images[i], (label_images[i].width, 20))
        draw = ImageDraw.Draw(combined_image)
        draw.text((10, 0), f"{label_files[i]}", fill=(255, 255, 255))
        draw.text((label_images[i].width + 10, 0), f"{output_files[i]}", fill=(255, 255, 255))
        combined_images.append(combined_image)

    # Save the combined images as a GIF
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    combined_images[0].save(os.path.join(output_dir, gif_name), format='GIF', save_all=True, append_images=combined_images[1:], duration=1500, loop=0)


