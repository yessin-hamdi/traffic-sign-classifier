import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from dataset import test_loader
from model import TrafficSignCNN

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TrafficSignCNN().to(device)
    model.load_state_dict(torch.load('models/V3_91.91.pth', map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix - Traffic Sign Classifier (v3)')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print('Saved confusion_matrix.png')

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    class_metrics = []
    for cls in range(43):
        cls_str = str(cls)
        if cls_str in report:
            class_metrics.append({
                'class': cls,
                'precision': report[cls_str]['precision'],
                'recall': report[cls_str]['recall'],
                'f1': report[cls_str]['f1-score'],
                'support': report[cls_str]['support'],
            })

    class_metrics.sort(key=lambda m: m['recall'])

    print('\nWorst 10 classes by recall:')
    print(f"{'Class':>6} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    for m in class_metrics[:10]:
        print(f"{m['class']:>6} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {m['support']:>8.0f}")

    print(f"\nOverall accuracy: {report['accuracy']:.4f}")
    print(f"Macro avg F1:     {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted avg F1:  {report['weighted avg']['f1-score']:.4f}")

    # --- Chasing the class-0 lead: what are the weakest classes actually
    # getting confused with? ---
    print('\nTop confusions for the 5 weakest classes:')
    worst_classes = [m['class'] for m in class_metrics[:5]]
    for cls in worst_classes:
        row = cm[cls].copy()
        true_positives = row[cls]
        row[cls] = 0  # exclude the diagonal (correct predictions) from the ranking
        top_confusions = row.argsort()[::-1][:3]  # top 3 highest remaining counts
        print(f'\nClass {cls} (correctly predicted {true_positives} times):')
        for conf_cls in top_confusions:
            if row[conf_cls] > 0:
                print(f'  -> predicted as class {conf_cls}: {row[conf_cls]} times')
