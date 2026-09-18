import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_data = datasets.GTSRB(root='data', split='train', download=True, transform=train_transform)
test_data = datasets.GTSRB(root='data', split='test', download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

if __name__ == '__main__':
    print(f'Training samples: {len(train_data)}')
    print(f'Test samples: {len(test_data)}')
    image, label = train_data[0]
    print(f'Image shape: {image.shape}, Label: {label}')
