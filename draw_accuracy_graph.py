import json, os

accuracy_list = json.load(open(os.path.join("D:\\result\\triplet", "accuracy.json")))
import matplotlib.pyplot as plt
print(accuracy_list)
plt.plot(accuracy_list)
plt.savefig("accuracy_graph.png")
