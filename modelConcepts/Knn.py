from operator import itemgetter
import numpy as np
'''
input example:
{
  "method": "predict_label",
  "features": [4.30936122, 4.28739283, 4.29680938, 4.33571647, 4.28774593],
  "k": 1
}
'''
def predict_label(examples, features, k, label_key="is_intrusive"):
    neighbors = find_k_nearest_neighbors(examples, features, k)

    intrusive_count = 0
    not_intrusive_count = 0

    for n in neighbors:

        if examples.get(n).get(label_key) == 0:
            not_intrusive_count += 1
        else:
            intrusive_count += 1

    return 1 if intrusive_count > not_intrusive_count else 0


def find_k_nearest_neighbors(examples, features, k):
    dist = dict()
    for key in examples:
        coor = examples.get(key).get('features')
        d = np.linalg.norm(np.array(coor) - np.array(features))
        dist[key] = d

    res = dict(sorted(dist.items(), key=itemgetter(1))[:k])

    return list(res.keys())
