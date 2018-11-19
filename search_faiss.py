import os
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/alexnet',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\\tfrecord',
                    help="Directory containing the dataset")
parser.add_argument('--embedding_size', default=128,
                    help="Directory containing the dataset")
parser.add_argument('--max_top_k', default=50,
                    help="Directory containing the dataset")
parser.add_argument('--restore_epoch', default=None,
                    help="Directory containing the dataset")

if __name__ == '__main__':

    args = parser.parse_args()

    query_embeddings = np.load(os.path.join(args.model_dir, "query_embeddings.npy")).astype(np.float32)
    index_embeddings = np.load(os.path.join(args.model_dir, "index_embeddings.npy")).astype(np.float32)
    query_labels = np.load(os.path.join(args.model_dir, "query_labels.npy"))
    index_labels = np.load(os.path.join(args.model_dir, "index_labels.npy"))

    import faiss

    ngpus = faiss.get_num_gpus()

    print("number of GPUs:", ngpus)
    cpu_index = faiss.IndexFlatL2(int(args.embedding_size))

    gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
        cpu_index
    )
    max_top_k = int(args.max_top_k)
    gpu_index.add(index_embeddings)  # add vectors to the index
    print(gpu_index.ntotal)

    k = max_top_k  # we want to see 4 nearest neighbors
    print("start search!")
    search_d, search_idx = gpu_index.search(query_embeddings, k)
    print("end search!")
    accuracy_list = []
    accuracies = []
    for i in range(max_top_k):
        accuracy_list.append([0, 0, 0])
    true_indices = []
    false_indices = []

    for i, q_label in enumerate(query_labels):
        searched_indices = index_labels[search_idx[i]]
        searched_distances = search_d[searched_indices]
        tmp_dist = [str(d) for d in search_d[i]]
        is_true = False
        for j in range(1, max_top_k + 1):
            tmp_indices = searched_indices[:j]
            if q_label in tmp_indices:
                accuracy_list[j - 1][0] += 1
                if not is_true:
                    true_indices.append([i, j, list(search_idx[i]), tmp_dist])
                    is_true = True
            else:
                accuracy_list[j - 1][1] += 1
        if not is_true:
            false_indices.append([i, list(search_idx[i]), tmp_dist])
    for accuracy in accuracy_list:
        accuracy[2] = float(accuracy[0]) / float(len(query_labels))
        accuracies.append(accuracy[2])
    # np.save(os.path.join(args.model_dir, "true_indices.npy"), true_indices)
    output_path = os.path.join(args.model_dir, "search_result")
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if args.restore_epoch is None:
        json.dump(true_indices, open(os.path.join(output_path, "true_indices.json"), "w+"))
        json.dump(false_indices, open(os.path.join(output_path, "false_indices.json"), "w+"))
        json.dump(accuracy_list, open(os.path.join(output_path, "accuracy_list.json"), "w+"))
        json.dump(accuracies, open(os.path.join(output_path, "accuracies.json"), "w+"))
    else:
        json.dump(true_indices, open(os.path.join(output_path, "%s_true_indices.json" % args.restore_epoch), "w+"))
        json.dump(false_indices, open(os.path.join(output_path, "%s_false_indices.json" % args.restore_epoch), "w+"))
        json.dump(accuracy_list, open(os.path.join(output_path, "%s_accuracy_list.json" % args.restore_epoch), "w+"))
        json.dump(accuracies, open(os.path.join(output_path, "%s_accuracies.json" % args.restore_epoch), "w+"))
    print(accuracy_list)
    print("top %s accuracy" % args.max_top_k,
          float(accuracy_list[int(args.max_top_k) - 1][0]) / float(len(query_labels)))
    # import matplotlib.pyplot as plt
    #
    # plt.plot(accuracies)
    # plt.savefig(os.path.join(args.model_dir, "accuracy_graph.png"))
