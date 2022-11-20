import argparse
import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm

#Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define the training function
def train(epoch, train_loader, model, criterion, optimizer, print_freq):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % print_freq == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

#Define the testing function returning test loss, accuracy, and confusion matrix
def test(test_loader, model, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = torch.zeros(model.fc.out_features, model.fc.out_features)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    #Extract class names from dataloader
    class_names = test_loader.dataset.classes
    #Print confusion matrix with class names and indicate which class is predicted
    print(tabulate.tabulate(confusion_matrix, headers=class_names, showindex=class_names, tablefmt="fancy_grid"))
    return test_loss, correct / len(test_loader.dataset), confusion_matrix

#Define the main function
def main():
    parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size of training')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--checkpoint_path', default='checkpoint.pth.tar', type=str, help='path to save the checkpoint')
    parser.add_argument('--train_path', default='.', type=str, help='path to the training data')
    parser.add_argument('--test_path', default='.', type=str, help='path to the testing data')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--image_size', default=224, type=int, help='image size')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay')
    parser.add_argument('--lr_decay_epoch', default=10, type=int, help='epoch number to decay the learning rate')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    args = parser.parse_args()

    #Define the model
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_classes)
    model = model.to(device)

    #Define the loss function and optimizer (ADAM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #Define the learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=args.lr_decay)

    #Define the data transforms
    train_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #Define the dataset
    train_dataset = torchvision.datasets.ImageFolder(root=args.train_path, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=args.test_path, transform=test_transforms)

    #Define the dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, model, criterion, optimizer, args.print_freq)
        test_loss, test_acc, confusion_matrix = test(test_loader, model, criterion)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'state_dict': model.state_dict(), 'class_names': train_dataset.classes}, args.checkpoint_path)
    print(f'Best accuracy: {best_acc:.2f}')


if __name__ == '__main__':
    main()