""" aux functions for play.py """

import matplotlib.pyplot as plt
import numpy as np
from ampligraph.latent_features.ComplEx import ComplEx
from ampligraph.latent_features.DistMult import DistMult
from ampligraph.latent_features.RandomBaseline import RandomBaseline
from ampligraph.latent_features.TransE import TransE
from sklearn import manifold
from sklearn.metrics import pairwise_distances


def rel_matrix(emb, rel_names, depth = 5):
    # compute the error btwn n-gen and 1-gen^n
    print('composition error analysis')
    for i in range(depth):
        dis = np.linalg.norm(np.power(emb[0], int(rel_names[i][0:])) - emb[i]) # prefix app is used
        print(rel_names[i], dis)
    print()

    print('nearby relation diff')
    for i in range(1, depth):
        diff = np.linalg.norm(emb[i] - emb[i-1])  # prefix app is used
        print(i, diff)

def annotate():
    # https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
    import matplotlib.pyplot as plt
    y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
    z = [0.15, 0.3, 0.45, 0.6, 0.75]
    n = [58, 651, 393, 203, 123]

    fig, ax = plt.subplots()
    ax.scatter(z, y)

    for i, txt in enumerate(n):
        ax.annotate(txt, (z[i], y[i]))
    plt.show()

def viz_distm(m, rs = 42, mode='mds', y=None, rel_names = None):
    """ Viz a distance matrix using MDS """
    # points = np.random.random((50, 2))
    # m = cdist(points, points, metric='euclidean')
    #TODO: add legend
    if mode == 'mds':
        mds = manifold.MDS(dissimilarity='precomputed', n_jobs=-1, random_state=rs, verbose=0)
        pos = mds.fit_transform(m)
    elif mode == 'mds_coordinate':
        distm = pairwise_distances(m,m, metric='euclidean')
        assert distm.shape == (m.shape[0], m.shape[0])
        mds = manifold.MDS(dissimilarity='precomputed', n_jobs=-1, random_state=rs, verbose=0)
        pos = mds.fit_transform(distm)
        # fake_names = list(range(pos.shape[0]))
        fake_names = rel_names

        fig, ax = plt.subplots()
        ax.scatter(pos[:, 0], pos[:, 1])

        for i, txt in enumerate(fake_names):
            ax.annotate(txt, (pos[i, 0], pos[i, 1]))
        plt.show()
        print('finish annotation')

    elif mode == 'tsne':
        tsne = manifold.TSNE(n_components=2,perplexity=100)
        pos = tsne.fit_transform(m)
    else:
        raise Exception('No such visualization mode')

    plt.scatter(pos[:, 0], pos[:, 1], c = y)
    plt.title('%s viz of matrix of size (%s %s)'%(mode, m.shape[0], m.shape[1]))
    plt.show()

def get_model(args):
    n_epoch = args.n_epoch
    if args.model == 'DistMul':
        model_args =    {'k': 400, 'epochs': n_epoch, 'eta': 50,
                        'loss': 'self_adversarial', 'loss_params': {'alpha': 1, 'margin': 1},
                        'regularizer': 'LP', 'regularizer_params': {'lambda': 0.0001, 'p': 2},
                        'optimizer': 'adam', 'optimizer_params': {'lr': 0.0005},'seed': 0,
                        'normalize_ent_emb': False, 'batches_count': 50}
    elif args.models == 'ComplEx':
        model_args =    {'batches_count': 10, 'seed': 0, 'epochs': n_epoch, 'k': 350, 'eta': 30,
                        'optimizer': 'adam', 'optimizer_params': {'lr': 0.0001},
                        'loss': 'self_adversarial', 'loss_params': {'margin': 0.5, 'alpha': 1},
                        'regularizer': 'LP', 'regularizer_params': {'p': 2, 'lambda': 1e-5},
                        'normalize_ent_emb': False}
    elif args.models == 'TransE':
        model_args =   {'k': 400, 'epochs': n_epoch, "eta": 50,
                        "loss": 'self_adversarial',
                        "loss_params": {'alpha': 1, 'margin': 1},
                        'regularizer': 'LP',
                        'regularizer_params': {'lambda': 0.0001, 'p': 2},
                        'optimizer': 'adam', 'optimizer_params': {'lr': 0.0005},
                        'seed': 0, 'normalize_ent_emb': False, 'batches_count': 50}
    else:
        model_args =  {'batches_count': 10, 'seed': 0, 'epochs': n_epoch, 'k': 150, 'eta': 10,
                       'optimizer': 'adam', 'optimizer_params': {'lr': 1e-3},
                       'loss': 'pairwise', 'loss_params': {'margin': 0.5, 'alpha': 1},
                       'regularizer': 'LP', 'regularizer_params': {'p': 2, 'lambda': 1e-5},
                       'normalize_ent_emb': False}

    common_model_args = {'verbose': 'args.verbose', 'fix_rel': 'args.fix_rel'}
    model_args = {**model_args, **common_model_args}

    if args.model == 'DistMul':
        model = DistMult(model_args)
    elif args.model == 'ComplEx':
        model = ComplEx(model_args)
    elif args.model == 'TransE':
        model = TransE(model_args)
    elif args.rb:
        model = RandomBaseline()
    else:
        raise Exception('No such model %s'%args.model)
    return model
