import sys
import os
sys.path.append("../")
sys.path.append("../core")
sys.path.append("../ml")
import json
from pathlib import Path
import run_retrofitting as run_retro
import multi_class_prediction as run_ml
from shutil import copyfile


def load_config(file_name):
    f = open(file_name, 'r')
    config = json.load(f)
    f.close()
    return config


def save_retro_config(retro_config):
    f = open("./config/retro_config.json", 'w')
    json.dump(retro_config, f, indent=2)
    f.close()


def run_retrofitting():
    run_retro.main(3, [
        './core/run_retrofitting.py',
        './config/retro_config.json',
        './config/db_config.json'
    ])


def run_ml_task(ml_config_file_name):
    run_ml.main(3, [
        './ml/multi_class_prediction.py',
        './config/db_config.json',
        './ml/'+ml_config_file_name
    ])


def save_results(result_path, ml_name, retro_name):
    files = ["./output/schema.json",
             "./output/schema.gml",
             "./output/food_cat_classify_plot_retro_vecs.png",
             "./output/food_cat_classify_retro_vecs.json",
             "./config/retro_config.json"]
    result_folder_path = result_path + "/" + ml_name + "/" + retro_name
    if os.path.exists(result_folder_path):
        if os.path.isdir(result_folder_path):
            print("WARNING: Result Folder already exists")
        else:
            print("WARNING: Result Folder is a File")
    else:
        Path(result_folder_path).mkdir(parents=True, exist_ok=True)

    for src_path in files:
        src = open(src_path, "r")
        dst_path = result_folder_path + "/" + os.path.basename(src.name)
        copyfile(src_path, dst_path)
        src.close()


def main(argc, argv):
    if argc < 2:
        print("Please specify runner config file")
        return

    runner_config = load_config(argv[1])
    os.chdir("../")

    for config in runner_config["configs"]:
        retro_config = config["retro"]
        retro_name = config["retro_name"]

        print("=========== Saving retro_config file for: %s ===========" % retro_name)
        save_retro_config(retro_config)
        print("=========== Running Retrofitting: %s ===========" % retro_name)
        run_retrofitting()

        for ml_config_file_name in config["ml"]:
            ml_config = load_config("./ml/"+ml_config_file_name)
            print("=========== Running ML Task: %s ===========" % ml_config["profile_name"])
            run_ml_task(ml_config_file_name)
            print("=========== Saving Results for: %s --> %s ===========" % (ml_config["profile_name"], retro_name))
            save_results(runner_config["result_path"], ml_config["profile_name"], retro_name)

    print("=========== FINISHED EXECUTION ===========")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
