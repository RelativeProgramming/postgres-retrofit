import sys
import os
sys.path.append("../")
import json

import matplotlib.pyplot as plt


def load_json(file_name):
    f = open(file_name, 'r')
    data = json.load(f)
    f.close()
    return data


def load_data(result_path, plots):
    result = []
    for plot in plots:
        path = result_path+plot+"food_cat_classify_retro_vecs.json"
        print("Reading JSON from: " + path)
        result.append(load_json(path)["retro_vecs"]["scores"])
    return result


def plot_graph(data, labels, title, y_axis, output_filename):
    fig1, ax1 = plt.subplots()
    ax1.set_title(title, fontsize=20)
    ax1.boxplot(data)
    plt.xticks(list(range(1, len(labels) + 1)), labels)
    plt.ylabel(y_axis, fontsize=18)
    plt.tick_params(labelsize=16)
    fig1.tight_layout()
    plt.savefig(output_filename)
    return


def main(argc, argv):
    if argc < 2:
        print("Please specify box_plotter config file")
        return

    config = load_json(argv[1])
    os.chdir("../")

    print("Loading data...")
    data = load_data(config["result_path"], config["plots"])
    print("Creating plot...")
    plot_graph(data, config["labels"], config["title"], config["y_axis"], "./output/custom_box_plot.png")
    print("Finished")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)