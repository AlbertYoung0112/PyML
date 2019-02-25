import numpy as np


class DecisionTree:
    class _Node:
        def __init__(self):
            self._parent = None
            self._child = {}
            self.node_data = {}

        def set_parent(self, parent_node):
            assert(type(parent_node) == type(self))
            self._parent = parent_node

        def append_child(self, child_name, child_node):
            assert(type(child_node) == type(self))
            assert(self.node_data["Node Type"] == "Branch")
            self._child[child_name] = child_node

        def find_child(self, child_name):
            if child_name in self._child:
                return self._child[child_name]
            else:
                return None

        def child_names(self):
            if self._child is None:
                return None
            return self._child.keys()

        def find_parent(self):
            return self._parent

    def _data_label_pair_validity_check(self, data, labels):
        assert(type(data) == np.ndarray)
        assert(type(labels) == np.ndarray)
        assert(data.ndim == 2)
        assert(labels.ndim == 2)
        assert(data.shape[1] == self._dims)
        assert(labels.shape[0] == data.shape[0])

    def __init__(self, dims):
        self._tree = None
        self._dims = dims

    def fit(self, data, labels, iterations):
        self._data_label_pair_validity_check(data, labels)

        def iteration(iter_data, iter_labels, dim_list):
            iter_node = self._Node()
            label_set = np.unique(iter_labels)
            if label_set.shape[0] == 1:
                iter_node.node_data = {"Node Type": "Leaf", "Class": label_set[0], "Count": iter_data.shape[0]}
                return iter_node
            if dim_list.shape[0] == 0 or np.where(iter_data != iter_data[0, :])[0].shape[0] == 0:
                label_count = np.array([[l, np.count_nonzero(iter_labels == l)] for l in label_set])
                iter_node.node_data = {
                    "Node Type": "Leaf",
                    "Class": label_count[np.argmax(label_count[:, 1]), 0],
                    "Count": label_count[np.argmax(label_count[:, 1]), 1]
                }
                return iter_node

            iter_node.node_data["Node Type"] = "Branch"
            info_gain = np.array([[dim, self._information_gain(iter_data, iter_labels, dim)] for dim in dim_list])
            dim_sel = info_gain[np.argmax(info_gain[:, 0]), 0].astype(np.int)
            dim_value = iter_data[:, dim_sel]
            dim_value_set = np.unique(dim_value)
            iter_node.node_data["Branch Dimension"] = dim_sel
            iter_node.node_data["Count"] = iter_data.shape[0]
            max_class = None
            max_count = -1
            for value in dim_value_set:
                next_iter_data_sel = iter_data[:, dim_sel] == value
                next_iter_data = iter_data[next_iter_data_sel, :]
                next_iter_label = iter_labels[next_iter_data_sel]
                if next_iter_data.shape[0] == 0:
                    next_node = self._Node()
                    label_count = np.array([[l, np.count_nonzero(iter_labels == l)] for l in label_set])
                    next_node.node_data = {
                        "Node Type": "Leaf",
                        "Class": label_count[np.argmax(label_count[:, 1]), 0],
                        "Count": 0
                    }
                    next_node.set_parent(iter_node)
                    iter_node.append_child(value, next_node)
                    continue
                else:
                    next_iter_dim_list = dim_list[dim_list != dim_sel]
                    next_node = iteration(next_iter_data, next_iter_label, next_iter_dim_list)
                    next_node.set_parent(iter_node)
                    iter_node.append_child(value, next_node)
                if next_node.node_data['Count'] > max_count:
                    max_count = next_node.node_data["Count"]
                    max_class = next_node.node_data["Class"]
            iter_node.node_data["Class"] = max_class
            return iter_node

        self._tree = iteration(data, labels, np.arange(0, data.shape[1]).astype(np.int))

    def predict(self, data):
        assert(type(data) == np.ndarray)
        assert(data.ndim == 2)
        assert(data.shape[1] == self._dims)
        assert(self._tree is not None)
        prediction = np.zeros((data.shape[0], 1))
        for index in range(data.shape[0]):
            prediction[index] = self._predict_one_data(data[index, :])
        return prediction

    def _predict_one_data(self, data):
        node = self._tree
        while node.node_data["Node Type"] != "Leaf":
            child = node.find_child(data[node.node_data['Branch Dimension']])
            if child is None:
                child_names = node.child_names()
                max_count = -1
                max_class = None
                for child_name in child_names:
                    child = node.find_child(child_name)
                    if child.node_data["Count"] > max_count:
                        max_class = child.node_data["Class"]
                        max_count = child.node_data["Count"]
                if max_class is None:
                    return 0
                return max_class
            else:
                node = child
        return node.node_data['Class']

    def _information_entropy(self, labels):
        label_set = np.unique(labels)
        label_count = np.array([np.count_nonzero(labels == l) for l in label_set])
        entropy = -1 * np.sum(label_count * np.log2(label_count))
        return entropy

    def _information_gain(self, data, labels, dim):
        dim_value_set = np.unique(data[:, dim])
        data_size = data.shape[0]
        entropy_gain = 0
        for dim_value in dim_value_set:
            dim_value_filter = data[:, dim] == dim_value
            selected_label = labels[dim_value_filter]
            entropy_temp = self._information_entropy(selected_label)
            entropy_gain -= selected_label.shape[0] / data_size * entropy_temp
        return entropy_gain

    def gini_index(self, data, labels):
        raise NotImplementedError

    def _pre_pruning(self):
        raise NotImplementedError

    def _post_pruning(self):
        raise NotImplementedError
