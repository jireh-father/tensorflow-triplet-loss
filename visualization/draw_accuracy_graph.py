import json, os

accuracy_list = json.load(open(os.path.join("D:\\result\\triplet", "accuracies.json")))
accuracy_list2 = json.load(open(os.path.join("D:\\result\\triplet", "train_accuracies.json")))
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

print(accuracy_list)
plt.plot(accuracy_list)
plt.plot(accuracy_list2)
plt.legend(['epoch 6', 'epoch 10'], loc='upper left')
plt.savefig("accuracy_graph.png")
