import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, os
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='d:/result/triplet',
                    help="Experiment directory containing params.json")
parser.add_argument('--epoch_list', default=None,
                    help="Directory containing the dataset")
if __name__ == '__main__':
    args = parser.parse_args()

    result_path = os.path.join(args.model_dir, "search_result")
    result_files = glob.glob(os.path.join(result_path, "*_accuracies.json"))
    assert len(result_files) > 0
    legends = []
    plt.figure(figsize=(8, 8))
    model_epochs = None
    if args.epoch_list is not None:
        model_epochs = [int(e) for e in args.epoch_list.split(",")]
    for result_file in result_files:
        i = int(os.path.basename(result_file).split("_")[0])
        if args.epoch_list is not None and i not in model_epochs:
            continue
        accuracies = json.load(open(result_file))
        plt.plot(accuracies)
        legends.append("epoch %d" % i)

    plt.legend(legends, loc='upper left')
    plt.savefig(os.path.join(args.model_dir, "search_result", "accuracy_graph.png"))
