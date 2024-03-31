import csv
from matplotlib.pylab import plt
import pandas as pd
import sys

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


def plot(experiment_name):
    # Read the data from the CSV file into a DataFrame
    df = pd.read_csv(f'/clusterfs/nvme/ethan/lls-defocus/experiments/{csv_file}')

    # Plot the data
    df.plot(y=["Training Loss", "Validation Loss"], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss of {experiment_name}')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python plot_function.py experiment_name csv_file")
        sys.exit(1)

    # Extract experiment_name and csv_file from command-line arguments
    experiment_name = sys.argv[1]
    csv_file = sys.argv[2]

    # Call the plot function
    plot(experiment_name, csv_file)