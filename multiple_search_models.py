import glob, os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/alexnet',
                    help="Experiment directory containing params.json")

if __name__ == '__main__':
    args = parser.parse_args()
    model_epochs = [os.path.basename(path).split("-")[1].split('.')[0] for path in
                    glob.glob(os.path.join(args.model_dir, "model*.index"))]
    model_epochs.sort()
    json.dump(model_epochs, open(os.path.join(args.model_dir, "search_epochs.json"), mode="w+"))
