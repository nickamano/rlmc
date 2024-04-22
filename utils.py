import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def list2txt(mylist, name):
    folder_path = "txt"
    file_name = name + ".txt"
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(file_path, "w") as file:
        for i in mylist:
            file.write(str(i) + "\n")


def plot1(scores, pretrain_episodes, average_n, name):
    average_scores = []
    temp = scores[pretrain_episodes - average_n: pretrain_episodes]
    for i in range(pretrain_episodes, len(scores)):
        idx = i - pretrain_episodes
        temp[idx % average_n] = scores[i]
        average_scores.append(sum(temp) / average_n)

    plt.plot(scores[pretrain_episodes:], label='score')
    plt.plot(average_scores, label='average score')
    plt.title(name)
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()
    # plt.savefig('')


def plot2(scores, scores_env, average_n, name):
    average_scores = []
    average_scores_env = []
    temp = []
    temp_env = []
    for i in range(len(scores)):
        if i < average_n:
            temp.append(scores[i])
            temp_env.append(scores_env[i])
        else:
            temp[i % average_n] = scores[i]
            temp_env[i % average_n] = scores_env[i]
        average_scores.append(np.mean(temp))
        average_scores_env.append(np.mean(temp_env))

    plt.plot(average_scores, label='average score after training')
    plt.plot(average_scores_env, label='average score from env')
    plt.title(name)
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()
    if not os.path.exists('plots'):
        os.mkdir('plots')

def visualize(positions, colors, name, xlim=(-5,10), ylim=(-5, 10)):
    # Setup the figure and axes...
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set(xlim=xlim, ylim=ylim)


def save_model(actor, name):
    model = actor
    folder_path = "pth"
    name = name + ".pth"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, name)
    torch.save(model, file_path)
