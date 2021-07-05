colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple'
for i, c in zip(label_set, colors):
    data = list(map(tuple, X_2d[all_labels == i]))
    ax_tsne[0].scatter(*zip(*data), c=c, label=i)
    ax_tsne[0].set_title('Cluster based')
    ax_tsne[0].legend()


