import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

class TrainingPlotter:
    def __init__(self, save_dir, class_names, filename="training_plot.png"):
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, filename)
        self.class_names = class_names
        
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": []
        }

    def update(self, train_loss, val_loss, train_acc, val_acc):
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)
        self._plot()

    def save_confusion_matrix(self, all_preds, all_labels, tag="best"):
        """
        Genera e salva la matrice di confusione.
        tag: può essere 'best' o 'final' per distinguere i file.
        """
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix ({tag})')
        
        save_path = os.path.join(self.save_dir, f"confusion_matrix_{tag}.png")
        plt.savefig(save_path)
        plt.close()
        
    def _plot(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        # Crea una figura con due subplot (Loss a sinistra, Accuracy a destra)
        plt.figure(figsize=(12, 5))
        
        # --- Grafico Loss ---
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss", marker='.')
        plt.plot(epochs, self.history["val_loss"], label="Val Loss", marker='.')
        plt.title("Loss Trend")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # --- Grafico Accuracy ---
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["train_acc"], label="Train Acc", marker='.')
        plt.plot(epochs, self.history["val_acc"], label="Val Acc", marker='.')
        plt.title("Accuracy Trend")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()