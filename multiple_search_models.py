import matplotlib

matplotlib.use('Agg')
import socket
import glob, os
import argparse
import util
import json
from datetime import datetime
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/alexnet',
                    help="Experiment directory containing params.json")
parser.add_argument('--embedding_size', default=128,
                    help="Directory containing the dataset")
parser.add_argument('--data_dir', default='D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\\tfrecord',
                    help="Directory containing the dataset")
parser.add_argument('--model_name', default="tfrecord",
                    help="Directory containing the dataset")
parser.add_argument('--max_top_k', default=20,
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
parser.add_argument('--notify_after_training', default='1',
                    help="Directory containing the dataset")
parser.add_argument('--use_old_model', default='0',
                    help="Directory containing the dataset")
parser.add_argument('--save_static_data', default="0",
                    help="Directory containing the dataset")
parser.add_argument('--num_preprocessing_threads', default=4,
                    help="Directory containing the dataset")

server_map = {"ip-172-31-12-89": "p3.2xlarge", "ip-172-31-29-214": "p3.8xlarge"}


def main(args, hostname):
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if args.epoch_list is None:
        model_epochs = [int(os.path.basename(path).split("-")[1].split('.')[0]) for path in
                        glob.glob(os.path.join(args.model_dir, "model*.index"))]
    else:
        model_epochs = [int(e) for e in args.epoch_list.split(",")]
    model_epochs.sort()
    for idx, i in enumerate(model_epochs):
        eval_cmd = 'python eval_retrieval_accuracy_v2.py --model_dir="%s" --data_dir="%s" --restore_epoch=%d --embedding_size=%d --model_name=%s --gpu_no=%s --image_size=%d --eval_batch_size=%d --preprocessing_name="%s" --use_old_model=%s --save_static_data=%s --num_preprocessing_threads=%d' % (
            args.model_dir, args.data_dir, i, int(args.embedding_size), args.model_name, args.gpu_no, int(
                args.image_size), int(args.eval_batch_size), args.preprocessing_name, args.use_old_model,
            (args.save_static_data if idx == 0 else "0"), int(args.num_preprocessing_threads))
        print(eval_cmd)
        os.system(eval_cmd)
        search_cmd = 'python search_faiss.py  --model_dir="%s" --data_dir="%s" --restore_epoch=%d --embedding_size=%d --max_top_k=%d --gpu_no=%s' % (
            args.model_dir, args.data_dir, i, int(args.embedding_size), int(args.max_top_k), args.gpu_no)
        print(search_cmd)
        os.system(search_cmd)

    import matplotlib.pyplot as plt

    result_path = os.path.join(args.model_dir, "search_result")
    legends = []
    max_acc = 0
    max_idx = None
    for i in model_epochs:
        accuracies = json.load(open(os.path.join(result_path, "%d_accuracies.json" % i)))
        if max_acc < accuracies[-1]:
            max_acc = accuracies[-1]
            max_idx = i
        plt.plot(accuracies)
        legends.append("%s %d" % (args.step_type, i))
    plt.legend(legends, loc='upper left')

    now = datetime.now().strftime('%Y%m%d%H%M%S')
    epochs_str = "all"
    if args.epoch_list is not None:
        epochs_str = "-".join(model_epochs)
    best_acc_filepath = os.path.join(args.model_dir, "search_result",
                                     "best_accuracy-date[%s]_model[%s]_log[%s]_data[%s]_embed[%d]_maxtopk[%d]_epochs[%s]_gpuno[%s].txt" % (
                                         now, args.model_name, os.path.basename(args.model_dir),
                                         os.path.basename(args.data_dir), int(args.embedding_size),
                                         int(args.max_top_k),
                                         epochs_str, args.gpu_no))
    best_acc_file = open(best_acc_filepath, mode="w+")
    best_acc_file.write("best accuracy: %f, best accuracy epoch: %d" % (max_acc, max_idx))
    best_acc_file.close()
    graph_filepath = os.path.join(args.model_dir, "search_result",
                                  "accuracy_graph-date[%s]_model[%s]_log[%s]_data[%s]_embed[%d]_maxtopk[%d]_epochs[%s]_gpuno[%s].png" % (
                                      now, args.model_name, os.path.basename(args.model_dir),
                                      os.path.basename(args.data_dir), int(args.embedding_size), int(args.max_top_k),
                                      epochs_str, args.gpu_no))
    plt.savefig(graph_filepath)
    if hostname[:3] == "ip-":
        os.system("aws s3 cp %s s3://igseo-ml-test-s3" % best_acc_filepath)
        os.system("aws s3 cp %s s3://igseo-ml-test-s3" % graph_filepath)
    if args.notify_after_training == "1":
        txt = "%s[%s]\n\n" % (hostname, socket.gethostbyname(socket.gethostname()))
        txt += "best accuracy %f at epoch %d\n\n" % (max_acc, max_idx)
        txt += "start time: %s\n" % start_time
        txt += "end time: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        txt += "\n[params]\n"
        for arg in vars(args):
            txt += "%s:%s\n" % (arg, str(getattr(args, arg)))
        util.send_msg_to_slack("\n\n==================================\nEvaluating is Done\n\n" + txt)


if __name__ == '__main__':

    try:
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hostname = socket.gethostname()
        if hostname in server_map:
            hostname = server_map[hostname] + "_" + hostname
        args = parser.parse_args()
        txt = "%s[%s]\n\n" % (hostname, socket.gethostbyname(socket.gethostname()))
        txt += "start time: %s\n" % start_time
        txt += "\n[params]\n"
        for arg in vars(args):
            txt += "%s:%s\n" % (arg, str(getattr(args, arg)))
        util.send_msg_to_slack("\n\n==================================\nStarted to evaluate!!!\n\n" + txt)
        main(args, hostname)
    except:
        txt = "%s[%s]\n\n" % (hostname, socket.gethostbyname(socket.gethostname()))
        txt += "start time: %s\n" % start_time
        txt += "end time: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        txt += "\n[stack trace]\n"
        txt += traceback.format_exc()
        txt += "\n[params]\n"
        for arg in vars(args):
            txt += "%s:%s\n" % (arg, str(getattr(args, arg)))
        util.send_msg_to_slack("\n\n==================================\nEvaluating Exception!!!\n\n" + txt)

    if args.shutdown_after_train == "1":
        os.system("sudo shutdown now")
