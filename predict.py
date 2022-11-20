import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

# Define the arguments
parser = argparse.ArgumentParser(description='PyTorch ResNet Prediction')
parser.add_argument('--checkpoint_path', default='checkpoint.pth.tar', type=str, help='path to load the checkpoint')
parser.add_argument('--image_path', default='.', type=str, help='path to the images to be predicted')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--image_size', default=224, type=int, help='image size')
parser.add_argument('--output_path', default='.', type=str, help='path to save the predictions')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
args = parser.parse_args() 

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, args.num_classes)
model = model.to(device)

# Define the data transforms
test_transforms = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Define the dataset
class TestDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.images = os.listdir(image_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = os.path.join(self.image_path, image)
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        name = self.images[idx]
        return image, name

# Define the dataloader
test_dataset = TestDataset(args.image_path, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

# Load the checkpoint
checkpoint = torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
class_names = checkpoint['class_names'] 

# Predict the classes of all images in the given folder
model.eval()
with torch.no_grad():
    # Store the predictions
    predictions = []
    for i, (images, names) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        # Save the image with a text stating the predicted class
        images = images[0].cpu().numpy()
        images = np.transpose(images, (1, 2, 0))
        images = (images - np.min(images)) / (np.max(images) - np.min(images))
        images = (images * 255).astype(np.uint8)
        image = Image.fromarray(images)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('arial.ttf', 25)
        predicted_class_str = class_names[predicted.item()]
        # Draw the predicted class label on the image 
        draw.text((5, 5), predicted_class_str, (0, 0, 0), font=font)
        image.save(os.path.join(args.output_path, names[0]))
        # Store the predictions
        predictions.append([names[0], predicted_class_str])

# Print the predictions
print(tabulate.tabulate(predictions, headers=['Image', 'Predicted Class'], tablefmt='fancy_grid'))