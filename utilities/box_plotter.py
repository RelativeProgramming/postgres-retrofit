import sys
import os
sys.path.append("../")
import json
from pathlib import Path

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


def create_output_dir(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            print("WARNING: Result Folder already exists")
        else:
            print("WARNING: Result Folder is a File")
    else:
        Path(path).mkdir(parents=True, exist_ok=True)


def plot_graph(data, labels, title, y_axis, fliers, output_filename):
    fig1, ax1 = plt.subplots()
    ax1.set_title(title, fontsize=20)
    if fliers:
        ax1.boxplot(data)
    else:
        ax1.boxplot(data, sym="")
    plt.xticks(list(range(1, len(labels) + 1)), labels)
    plt.ylabel(y_axis, fontsize=18)
    plt.tick_params(labelsize=16)
    fig1.tight_layout()
    plt.savefig(output_filename)
    return


def save_plotter_config(config, path):
    f = open(path + "/box_plotter_config.json", 'w')
    json.dump(config, f, indent=2)
    f.close()


def main(argc, argv):
    if argc < 2:
        print("Please specify box_plotter config file")
        return

    config = load_json(argv[1])
    os.chdir("../")

    print("Loading data...")
    data = load_data(config["result_path"], config["plots"])
    print("Creating plot...")
    create_output_dir(config["output_path"]+config["name"])
    plot_graph(data, config["labels"], config["title"], config["y_axis"], config["fliers"],
               config["output_path"]+config["name"]+"/plot.png")
    save_plotter_config(config, config["output_path"]+config["name"])
    print("Finished")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)