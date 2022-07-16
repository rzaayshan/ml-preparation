import random
import numpy as np

'''

user_feature_map:
{
"uid_0":[-1.23, -2.57, -1.78, -0.56],
"uid_1":[-2.54, -1.82, -1.05, -1.96],
...
}
input:
{
  "k": 1
}
output:
[
  [-1.0659, -1.0981, -1.0667, -1.0846]
]
'''

class Centroid:
    def __init__(self, location):
        self.location = location
        self.closest_users = set()


def get_k_means(user_feature_map, num_features_per_user, k):
    # Don't change the following two lines of code.
    random.seed(42)

    # Gets the inital users, to be used as centroids.
    inital_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)

    centroids = []

    for centroid_user in inital_centroid_users:
        centroids.append(Centroid(user_feature_map[centroid_user]))

    for _ in range(10):
        for node_key in user_feature_map:
            min_centroid = -1
            min_dist = np.inf

            for i in range(len(centroids)):
                node_loc = np.array(user_feature_map[node_key])
                center_loc = np.array(centroids[i].location)
                manhattan_dist = sum(abs(node_loc - center_loc))

                if min_dist > manhattan_dist:
                    min_centroid = i
                    min_dist = manhattan_dist
            centroids[min_centroid].closest_users.add(node_key)

        for centroid in centroids:
            centroid.location = np.sum([user_feature_map[user] for user in centroid.closest_users], axis=0) / len(centroid.closest_users)
            centroid.closest_users = set()

    return [centroid.location for centroid in centroids]








