import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/alexnet',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\\tfrecord',
                    help="Directory containing the dataset")
parser.add_argument('--embedding_size', default=512,
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

    gpu_index.add(index_embeddings)  # add vectors to the index
    print(gpu_index.ntotal)

    k = 20  # we want to see 4 nearest neighbors
    print("start search!")
    search_d, search_idx = gpu_index.search(query_embeddings, k)
    print("end search!")
    true_cnt = 0
    false_cnt = 0
    for i, q_label in enumerate(query_labels):
        searched_indices = index_labels[search_idx[i]]
        print(searched_indices)
        if q_label in searched_indices:
            true_cnt += 1
        else:
            false_cnt += 1
    print("true", true_cnt, "false", false_cnt)
