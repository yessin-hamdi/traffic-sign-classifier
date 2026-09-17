import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_data = datasets.GTSRB(root='data', split='train', download=True, transform=transform)
test_data = datasets.GTSRB(root='data', split='test', download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

if __name__ == '__main__':
    print(f'Training samples: {len(train_data)}')
    print(f'Test samples: {len(test_data)}')
    image, label = train_data[0]
    print(f'Image shape: {image.shape}, Label: {label}')