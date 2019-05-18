import pickle
import matplotlib.pyplot as plt

def loss_plot(history, save=False):
    loss = history["loss"]
    val_loss = history["val_loss"]
    plt.plot(loss, color="blue")
    plt.plot(val_loss, color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(("Training", "Validation"))
    if save: plt.savefig("../output/plot")
    plt.show()


if __name__ == '__main__':
    filepath = r"../output/history_Mon_Dec_17_06~55~45_2018"
    file = open(filepath, "rb")
    history = pickle.load(file)
    print(history.keys())
    loss_plot(history, save=True)