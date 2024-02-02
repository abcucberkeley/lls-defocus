import csv
from matplotlib.pylab import plt

def plot_loss(experiment_name):
    epochs = []
    train_loss = []
    val_loss = []

    with open(f'../experiments/{experiment_name}.csv') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for i, row in enumerate(reader):
            epochs.append(i+1)
            train_loss.append(float(row[0]))
            val_loss.append(float(row[1]))
    
    print(epochs)
    print(train_loss)
    print(val_loss)
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()
    plt.savefig(f'../experiments/{experiment_name}.png')

plot_loss("test-001")