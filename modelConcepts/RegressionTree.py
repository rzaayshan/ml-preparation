import numpy as np
import math


class TreeNode:
    def __init__(self, examples):
        self.examples = examples
        self.left = None
        self.right = None
        self.split_point = None

    def split(self):
        if len(self.examples) == 1:
            return

        left = None
        right = None
        best_split = None
        mse = math.inf
        for key in self.examples[0].keys():

            if key == 'bpd':
                continue

            self.examples.sort(key=lambda example: example[key])
            data = self.examples

            for i in range(len(self.examples) - 1):
                avg = (data[i][key] + data[i + 1][key]) / 2
                l = []
                r = []
                for d in data:
                    if d[key] <= avg:
                        l.append(d)
                    else:
                        r.append(d)

                label_l = [k['bpd'] for k in l]
                label_r = [k['bpd'] for k in r]
                avg_l = sum(label_l) / len(l)
                avg_r = sum(label_r) / len(r)
                label_l = np.array(label_l)
                label_r = np.array(label_r)

                mse_l = sum((label_l - avg_l) ** 2) / len(label_l)
                mse_r = sum((label_r - avg_r) ** 2) / len(label_r)
                mse_temp = (mse_l * len(label_l) + mse_r * len(label_r)) / len(data)

                if mse_temp < mse:
                    mse = mse_temp
                    best_split = {'key': key, 'avg': avg}
                    left = l
                    right = r

        self.split_point = best_split

        self.left = TreeNode(left)
        self.left.split()

        self.right = TreeNode(right)
        self.right.split()


class RegressionTree:
    def __init__(self, examples):
        self.root = TreeNode(examples)
        self.train()

    def train(self):
        self.root.split()

    def predict(self, example):
        node = self.root

        while node.left or node.right:

            if example[node.split_point['key']] <= node.split_point['avg']:
                if node.left:
                    node = node.left
            else:
                if node.right:
                    node = node.right
        leafs = [k['bpd'] for k in node.examples]
        avg = sum(leafs) / len(leafs)
        return avg




