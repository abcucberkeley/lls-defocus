import csv
from matplotlib.pylab import plt

def plot_loss(experiment_name, epoch):
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
    
    plt.plot(epochs, train_loss, label="Training Loss", color="blue")
    plt.plot(epochs, val_loss, label="Validation Loss", color="red")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    # if epoch % 10 == 0 and epoch != 0:
    #     plt.show()
    plt.savefig(f'../experiments/{experiment_name}.png')