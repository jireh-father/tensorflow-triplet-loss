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
parser.add_argument('--epoch_list', default=None,
                    help="Directory containing the dataset")
parser.add_argument('--shutdown_after_train', default="0",
                    help="Directory containing the dataset")
parser.add_argument('--gpu_no', default="0",
                    help="Directory containing the dataset")
parser.add_argument('--step_type', default="epoch",
                    help="Directory containing the dataset")
parser.add_argument('--image_size', default=128,
                    help="Directory containing the dataset")
parser.add_argument('--eval_batch_size', default=256,
                    help="Directory containing the dataset")
parser.add_argument('--preprocessing_name', default='None',
                    help="Directory containing the dataset")

if __name__ == '__main__':
    args = parser.parse_args()
    if args.epoch_list is None:
        model_epochs = [int(os.path.basename(path).split("-")[1].split('.')[0]) for path in
                        glob.glob(os.path.join(args.model_dir, "model*.index"))]
    else:
        model_epochs = [int(e) for e in args.epoch_list.split(",")]
    model_epochs.sort()
    for i in model_epochs:
        eval_cmd = 'python eval_retrieval_accuracy_v2.py --model_dir="%s" --data_dir="%s" --restore_epoch=%d --embedding_size=%d --model_name=%s --gpu_no=%s --image_size=%d --eval_batch_size=%d --preprocessing_name=%s' % (
            args.model_dir, args.data_dir, i, int(args.embedding_size), args.model_name, args.gpu_no, int(
                args.image_size), int(args.eval_batch_size), args.preprocessing_name)
        print(eval_cmd)
        os.system(eval_cmd)
        search_cmd = 'python search_faiss.py  --model_dir="%s" --data_dir="%s" --restore_epoch=%d --embedding_size=%d --max_top_k=%d --gpu_no=%s' % (
            args.model_dir, args.data_dir, i, int(args.embedding_size), int(args.max_top_k), args.gpu_no)
        print(search_cmd)
        os.system(search_cmd)

    import matplotlib.pyplot as plt

    result_path = os.path.join(args.model_dir, "search_result")
    legends = []
    for i in model_epochs:
        accuracies = json.load(open(os.path.join(result_path, "%d_accuracies.json" % i)))
        plt.plot(accuracies)
        legends.append("%s %d" % (args.step_type, i))
    plt.legend(legends, loc='upper left')
    from datetime import datetime

    now = datetime.now().strftime('%Y%m%d%H%M%S')
    epochs_str = "all"
    if args.epoch_list is not None:
        epochs_str = "-".join(model_epochs)

    plt.savefig(os.path.join(args.model_dir, "search_result",
                             "accuracy_graph-date[%s]_model[%s]_log[%s]_data[%s]_embed[%d]_maxtopk[%d]_epochs[%s]_gpuno[%s].png" % (
                                 now, args.model_name, os.path.basename(args.model_dir),
                                 os.path.basename(args.data_dir), int(args.embedding_size), int(args.max_top_k),
                                 epochs_str, args.gpu_no)))
    if args.shutdown_after_train == "1":
        os.system("sudo shutdown now")
