import matplotlib.pyplot as plt
import random
from collections import Counter
from dataset import train_data

random.seed(42)
sample_indices = random.sample(range(len(train_data)), 10)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, idx in zip(axes.flat, sample_indices):
    image, label = train_data[idx]
    img_np = image.permute(1, 2, 0).numpy()
    ax.imshow(img_np)
    ax.set_title(f'Class {label}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('sample_images.png')
print('Saved sample_images.png')

labels = [label for _, label in train_data._samples]
counts = Counter(labels)
print(f'Number of classes: {len(counts)}')
print(f'Smallest class: {min(counts.values())} images')
print(f'Largest class: {max(counts.values())} images')