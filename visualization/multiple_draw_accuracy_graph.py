import matplotlib.pyplot as plt
import matplotlib
import glob, os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/alexnet',
                    help="Experiment directory containing params.json")

if __name__ == '__main__':
    args = parser.parse_args()

    matplotlib.use('Agg')
    result_path = os.path.join(args.model_dir, "search_result")
    result_files = glob.glob(os.path.join(result_path, "*_accuracies.json"))
    legends = []
    for result_file in result_files:
        i = int(os.path.basename(result_file).split("_")[0])
        accuracies = json.load(open(result_file))
        plt.plot(accuracies)
        legends.append("epoch %d" % i)
    plt.legend(legends, loc='upper left')
    plt.savefig("accuracy_graph.png")
