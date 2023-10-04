import numpy as np

path = './data/mutag_samples/mutag_sample_weights.npy'
# path = './data/mutag/sample_weights.npy'

data0 = np.load(path,allow_pickle=True)
print()

new_data = []
for i in range(data0.shape[0]):
    temp_data = data0[i]
    temp_list = []
    for c in range(temp_data.shape[0]):
        l = temp_data[c]
        t_l = []
        for s in l:
            t_l.append(list(s))
        temp_list.append(t_l)
    new_data.append(temp_list)

np.save('./data/mutag_samples/sample_weights_lists.npy',new_data)

exit(0)

from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth

dataset_name = 'mutag'
explanation_labels, indices = load_dataset_ground_truth(dataset_name)
path = './data/mutag/sample_weights_100_1000.npy'
data0 = np.load(path,allow_pickle=True)
new_data = []

edit_distance_lists = [[], [], [], [], [], [], []]
for i in range(len(explanation_labels[0])):
    if i in indices:
        idx = indices.index(i)
        new_data.append(data0[idx])
    else:
        new_data.append(edit_distance_lists)

np.save('./data/mutag/sample_weights.npy',new_data)



path = './data/ba2motifs/sample_weights_0_100.npy'
path1 = './data/ba2motifs/sample_weights_100_1000.npy'

data0 = np.load(path,allow_pickle=True)
data1 = np.load(path1,allow_pickle=True)
print("xx")
new_data = np.concatenate([data0,data1],axis=0)
np.save('./data/ba2motifs/sample_weights.npy',new_data)
