import glob, os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/alexnet',
                    help="Experiment directory containing params.json")
parser.add_argument('--embedding_size', default=128,
                    help="Directory containing the dataset")
parser.add_argument('--data_dir', default='D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\\tfrecord',
                    help="Directory containing the dataset")
parser.add_argument('--model_name', default="tfrecord",
                    help="Directory containing the dataset")
parser.add_argument('--max_top_k', default=50,
                    help="Directory containing the dataset")

if __name__ == '__main__':
    args = parser.parse_args()
    model_epochs = [int(os.path.basename(path).split("-")[1].split('.')[0]) for path in
                    glob.glob(os.path.join(args.model_dir, "model*.index"))]
    model_epochs.sort()
    for i in model_epochs:
        eval_cmd = "python eval_retrieval_accuracy_v2.py --model_dir=%s --data_dir=%s --restore_epoch=%d --embedding_size=%d --model_name=%s" % (
            args.model_dir, args.data_dir, i, int(args.embedding_size), args.model_name)
        print(eval_cmd)
        os.system(eval_cmd)
        search_cmd = "python search_faiss.py  --model_dir=%s --data_dir=%s --restore_epoch=%d --embedding_size=%d --max_top_k=%d" % (
            args.model_dir, args.data_dir, i, int(args.embedding_size), int(args.max_top_k))
        print(search_cmd)
        os.system(search_cmd)

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')
    result_path = os.path.join(args.model_dir, "search_result")
    legends = []
    for i in model_epochs:
        accuracies = json.load(open(os.path.join(result_path, "%d_accuracies.json" % i)))
        plt.plot(accuracies)
        legends.append("epoch %d" % i)
    plt.legend(legends, loc='upper left')
    plt.savefig("accuracy_graph.png")
