import json, os
# json.dump(accuracy_list, open(os.path.join(args.model_dir, "accuracy_list.json"), "w+"))
# accuracy_list = json.load(open(os.path.join(args.model_dir, "accuracy_list.json")))
# print(accuracy_list)
# print("top %s accuracy" % args.top_k, float(accuracy_list[int(args.top_k) - 1][0]) / float(len(query_labels)))
import matplotlib.pyplot as plt

plt.plot([3.0,4.0,5.0])
plt.savefig("accuracy_graph.png")
