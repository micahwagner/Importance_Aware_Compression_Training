import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def test_best_model(model_path, model, out_dir, device, train_loader, test_loader, criterion, losses, test_losses, epochs, best_epoch):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    dataset = getattr(train_loader, 'dataset', None)
    mode = getattr(dataset, 'compression_mode', 'unknown')
    file_name = os.path.join(out_dir, "summary")

    def evaluate(dataloader):
        correct = total = count = 0
        total_loss = 0.0
        with torch.no_grad():
            for imgs, labels, global_indices in dataloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels).cpu().detach().item()
                total_loss += loss
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                count += 1
        accuracy = 100 * correct / total
        avg_loss = total_loss / count
        return avg_loss, accuracy

    train_loss, train_acc = evaluate(train_loader)
    test_loss, test_acc = evaluate(test_loader)

    print("\n--- Best Model Evaluation ---")
    print(f"Best Epoch: {best_epoch}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Testing  Loss: {test_loss:.4f}")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Testing  Accuracy: {test_acc:.2f}%")

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name + ".txt", 'w') as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Training Loss: {train_loss:.4f}\n")
        f.write(f"Testing  Loss: {test_loss:.4f}\n")
        f.write(f"Training Accuracy: {train_acc:.2f}%\n")
        f.write(f"Testing  Accuracy: {test_acc:.2f}%\n")
        f.write(f"Compression method: {mode}\n")
        if mode == "fixed":
            f.write(f"Fixed quality: q{dataset.fixed_quality}\n")
        elif mode == "manual":
            f.write(f"Manual thresholds: {dataset.manual_thresholds}\n")

    # Save plot
    plt.plot(range(1, epochs+1), losses, linestyle='--', label='Training Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Testing Loss')
    legend = plt.legend(fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.title(f'ResNet18 on CIFAR-10')
    plt.savefig(file_name + ".pdf")
    np.savez(
        os.path.join(out_dir, "loss_data.npz"),
        train=np.array(losses),
        val=np.array(test_losses)
    )
