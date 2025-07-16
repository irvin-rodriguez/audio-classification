import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_loss(train_losses, val_losses, model_name):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss Over Epochs for {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./outputs/{model_name}_loss.png")
    plt.close()

def plot_accuracy(train_accuracies, val_accuracies, model_name):
    plt.figure(figsize=(6, 4))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"Accuracy Over Epochs for {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./outputs/{model_name}_accuracy.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, normalize='true', values_format=".2f")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"./outputs/{model_name}_normalized_confusion_matrix.png")
    plt.close()
    
    print(f"Saved {model_name} confusion matrix.")