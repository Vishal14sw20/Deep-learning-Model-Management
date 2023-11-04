import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import cycle

matplotlib.use('Agg')  # Set the backend to TkAgg

width = 0.08
gap = 0.12  # Adjust the gap between groups
#plt.figure(figsize=(15, 18))

models = ['Resnet', 'alexnet', 'vgg']

#data = {'original': [42.74, 217.62], 'zlib': [34.30, 0.22], 'gzip': [34.30, 0.22]}
data2 = {'original': [42, 491, 333], 'zlib': [34, 395, 253], 'gzip': [34, 395, 253],
         'lzma': [30, 375, 239], 'delta': [44, 491, 333], 'rle': [55, 575, 400]}


def size_plot(path, operator, dataset, extraction, models, plt_dict, type):
    fig, ax = plt.subplots(figsize=(11, 7))
    chart_name = path + "plots/{}/{}_{}_{}_bar_plot.png".format(operator, dataset, extraction, type)
    # Create x values for each group
    x = np.arange(len(models))

    # Create a legend
    legend_labels = list(plt_dict.keys())


    # Define your list of hardcoded colors
    #hardcoded_colors = ['darkblue', 'olive', 'green', 'teal', 'purple', 'maroon']
    hardcoded_colors = ['darkblue', 'olive', 'green', 'teal', 'purple', 'maroon', 'red']

    # Plot data in grouped manner of bar type using hardcoded colors
    for i, (label, values) in enumerate(plt_dict.items()):
        ax.bar(x + i * gap, values, width, color=hardcoded_colors[i], label=label)


    #ax.set_xticks(x)
    ax.set_xticks(x + (len(legend_labels) - 1) * gap / 2)
    ax.set_xlabel("Model", fontsize=16)
    if type == 'Size':
        ax.set_ylabel("Size in MB")
    elif type == 'TTS':
        ax.set_ylabel("TTS")
    elif type == 'TTR':
        ax.set_ylabel("TTR", fontsize=16)
    ax.set_xticklabels(models)
    ax.legend(legend_labels)


    # Display numbers in the vertical bars with an offset
    for i in range(len(models)):
        for label, values in plt_dict.items():
            ax.text(x[i] + (list(plt_dict.keys()).index(label) * gap), values[i], str(values[i]), ha='center', va='bottom')

    if extraction == 'fe' or type == 'TTS' or type == 'TTR':
        ax.set_yscale('log')
    plt.title("{} of Models after Compression".format(type), fontsize=20)
    plt.savefig(chart_name)

def size_plot2(path, operator, dataset, extraction, models, plt_dict, type):
    big_fig_size = (10, 6.5)
    fig, axes = plt.subplots(1, len(models), figsize=(big_fig_size[0], big_fig_size[1]))

    chart_name = os.path.join(path, "plots", f"{operator}/{dataset}_{extraction}_{type}_bar_plot.png")

    for i, model in enumerate(models):
        ax = axes[i]
        x = np.arange(len(plt_dict))
        values = [plt_dict[method][i] for method in plt_dict]
        hardcoded_colors = ['darkblue', 'olive', 'green', 'teal', 'purple', 'maroon', 'red']

        # Adjust the width to reduce the gap between bars
        bar_width = 0.60

        bars = ax.bar(x, values, bar_width, color=hardcoded_colors)

        #ax.get_xaxis().set_visible(False)  # Hide the x-axis

        ax.set_xticks(x)  # Set the x-ticks at the positions of the bars
        ax.set_xticklabels(list(plt_dict.keys()), rotation=45)
        ax.set_xlabel("", fontsize=16)  # Clear x-axis label
        ax.set_ylabel("Size in MB" if i == 0 and type == 'Size' else '', fontsize=16)
        ax.set_title(model, fontsize=16)  # Set the model name as a title

        for j in range(len(plt_dict)):
            ax.text(x[j], values[j], str(values[j]), ha='center', va='bottom')

        if extraction == 'fe' or type == 'TTS' or type == 'TTR':
            ax.set_yscale('log')

    # Set the size of the big figure
    fig.set_size_inches(big_fig_size[0], big_fig_size[1])


    plt.tight_layout()

    # Save the combined figure
    plt.savefig(chart_name)




#size_plot('/Users/vishalkumarlohana/MyProjects/Thesis/', 'd', 'ft', models, data2)


