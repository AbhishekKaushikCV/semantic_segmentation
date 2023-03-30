import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import batch_size, learning_rate, num_epochs, in_channels, num_classes, feats_dir, labels_dir, images_dir
from model import SegmentationModel
from datasets import KittiDataset
from utils import train_model, masks2labels, color_segmap, plot_training_results, create_gif

def main() -> None:
    """
    Trains the segmentation model using the Kitti dataset.

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    train_dataset = KittiDataset(feats_dir, labels_dir, num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the model
    model = SegmentationModel(in_channels, num_classes)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_train_loss, total_pixel_acc = [], []
    # Train the model
    for epoch in tqdm(range(num_epochs)):
        train_loss, train_pixel_acc = train_model(model, train_loader, optimizer, criterion, device, masks2labels, color_segmap)
        total_train_loss.append(train_loss)
        total_pixel_acc.append(train_pixel_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Pixel Acc: {train_pixel_acc:.4f}')

    plot_training_results(total_train_loss, total_pixel_acc)

    create_gif(image_dir=images_dir, output_dir=images_dir, gif_name='GIF.gif'  )

if __name__ == '__main__':
    main()
