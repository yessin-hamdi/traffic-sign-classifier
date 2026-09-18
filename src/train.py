import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import train_loader, test_loader
from model import TrafficSignCNN

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = TrafficSignCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 80
    patience = 5
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}: train loss={epoch_loss:.4f}, test accuracy={test_accuracy:.2f}%')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f'  New best: {best_accuracy:.2f}% -- checkpoint saved to models/best_model.pth')
        else:
            patience_counter += 1
            print(f'  No improvement. Patience: {patience_counter}/{patience}')
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}.')
                break

    print(f'Training complete. Best test accuracy: {best_accuracy:.2f}%')
