import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids 

def heatmap(Distance,name,ax=None):
    if ax is None:
        ax = plt.gca()

    sns.heatmap(
        Distance,
        annot = None,
        cmap = "YlOrRd",
        linewidths = 0.5,
        xticklabels = Distance.columns,
        yticklabels = Distance.columns,
        ax = ax
    )

    ax.set_title(f"{name} Heatmap", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


def cluster(Distance,k):
# k: unique terms in origianl equations
    kmedoids = KMedoids(
        n_clusters = k,
        metric = 'precomputed',
        method = 'pam',
        init = 'build',
        max_iter = 300,
        random_state = 27
    )
    kmedoids.fit(Distance)
    labels = kmedoids.labels_
    medoids = kmedoids.medoid_indices_

    func_names = Distance.columns
    medoid_funcs = [func_names[i] for i in medoids]
    clusters = {c: [] for c in range(kmedoids.n_clusters)}
    for i,lab in enumerate(labels):
        clusters[lab].append(func_names[i])
    return clusters, medoid_funcs