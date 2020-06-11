import matplotlib.pyplot as plt
import os


def DrawPics(features, labels, epoch):
    plt.clf()
    color = ['#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF',
             '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF',
             ]
    for i in range(10):
        plt.scatter(features[labels == i, 0], features[labels == i, 1], color=color[i])
    plt.legend(["0","1","2","3","4","5","6","7","8","9"], loc = 'upper right')
    plt.title(f"Epoch-{epoch}")
    if os.path.exists("./Pics") is False:
        os.mkdir("./Pics")
    plt.savefig(f"Pics/Epoch-{epoch}")



