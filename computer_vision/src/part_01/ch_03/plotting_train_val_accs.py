import matplotlib.pyplot as plt
import json

with open("results_accuracies.json", "r") as f:
    results = json.load(f)

train_accs = results["train_accs"]
val_accs = results["val_accs"]
test_accs = results["test_accs"]
epochs = range(1, len(train_accs)+1)

plt.plot(epochs,train_accs, 'b-', label="Train Acc")
plt.plot(epochs,val_accs, 'g--', label="Val Acc")

# plt.plot(epochs,test_accs, 'r-', label="Test Acc")
if len(test_accs) == 1:
    plt.axhline(y=test_accs[0], color='r', linestyle=':', 
                label=f"Test Acc: {test_accs[0]:.4f}", linewidth=3, alpha=0.7)
else:
    plt.plot(epochs[-len(test_accs):], test_accs, 'r-', label="Test Acc", marker='^')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('mini_alexnet_100_epochs.png')
# plt.show()
