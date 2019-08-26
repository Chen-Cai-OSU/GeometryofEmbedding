import matplotlib
matplotlib.use('tkagg')
import os
import subprocess
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import tensorflow as tf
from ampligraph.common.aux import rel_rank_stat, load_data, eigengap
from ampligraph.common.aux_play import get_model, viz_distm
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score

from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
import warnings
from pymongo import MongoClient
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR) # https://stackoverflow.com/questions/48608776/how-to-suppress-tensorflow-warning-displayed-in-result

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--depth", default=8, type=int, help='method')
parser.add_argument('--rb', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--fix_rel', action='store_true')
parser.add_argument("--viz", action='store_true')

parser.add_argument('--data', type = str, default='single_fam_tree')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--prob', type=float, default=0)
parser.add_argument("--seed", default=42, type=int, help='random seed')
parser.add_argument("--add", default=1, type=int, help='additional relation')
parser.add_argument("--extra_rel", default=30, type=int, help='extra relation')
parser.add_argument("--noise_rel", default=1, type=int, help='noise relation (half noise)')
parser.add_argument("--n_node", default=1000, type=int, help='n_nodes')
parser.add_argument("--model", default='ComplEx', type=str, help='model name')
parser.add_argument("--period", default=3, type=int, help='the period of relations')

client = MongoClient('localhost', 27017)
EXPERIMENT_NAME = 'KG_corrupt'
YOUR_CPU = None
DATABASE_NAME = 'KG_corrupt'
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))
warnings.filterwarnings("ignore", category=DeprecationWarning)
from ampligraph.common.aux import timefunction

# @timefunction
def topk_tails(model, hr=np.array(['101', '1']), top_k=5, print_f = False):
    """
    :param model:
    :param hr: h and r
    :param top_k:
    :return:
    """
    # hr = np.array([['101', '1']])
    hr = hr.reshape(1, 2)
    C = np.array([np.append(hr, x) for x in list(model.ent_to_idx.keys())])
    tmp = np.array(model.predict(C)).reshape(C.shape[0], 1) # np.array of shape (n,1)
    df = pd.DataFrame(np.hstack([C, tmp]), columns=['head', 'relation', 'tail_', 'score'])

    def filter_score(row):
        if float(row.score) < 1e-4:
            return 0
        else:
            return float(row.score)

    df['score_filter'] = df.apply(filter_score, axis=1)
    df = df.sort_values('score_filter', ascending=False, inplace=False)

    if print_f:
        print('Top %s solutions for query (%s, %s, ?)' % (top_k, hr[0, 0], hr[0, 1]))
        print(df[:top_k])
        print('-' * 50)

    # print(df.tail_)
    best_tails = list(df.tail_) # return the best tail

    return int(best_tails[0]), int(best_tails[1])

def ranks_summary(ranks):
    mrr = mrr_score(ranks)
    hits_1, hits_3, hits_10, hits_20, hits_50 = hits_at_n_score(ranks, n=1), hits_at_n_score(ranks,
                                                                                             n=3), hits_at_n_score(
        ranks, n=10), hits_at_n_score(ranks, n=20), hits_at_n_score(ranks, n=50)
    print("MRR: %f, Hits@1: %f, Hits@3: %f, Hits@10: %f, Hits@20: %f, Hits@50: %f" % (
        mrr, hits_1, hits_3, hits_10, hits_20, hits_50))
    print('-' * 150)

def eval_corrution(model, filter, args, x, noise = 1, noise_loc = 't'):
    corrupt_test = corrupt(x, period=args.n_node, noise=noise, corrput_location= noise_loc)
    args_ = {'strict':False, 'model': model, 'filter_triples': filter, 'verbose': args.verbose, 'use_default_protocol': True}
    ranks_corrupt = evaluate_performance(corrupt_test, **args_)
    for i in range(10):
        print('noise ' + str(noise) + ' at ' + noise_loc, corrupt_test[i], ranks_corrupt[2*i], ranks_corrupt[2*i+1])
    ranks_summary(ranks_corrupt)
    print('-'*150)

def corrupt(x, period = 1000, noise = 1, corrput_location = 'h'):
    # array of shape (n, 3)
    n, d = x.shape
    assert d == 3
    res = []
    for i in range(n):
        h, r, t = x[i]

        if corrput_location == 't':
            t = str((int(t) + noise)%period)  # 1 is noise here
        elif corrput_location == 'r':
            # todo make sure no new relation is introduced
            r = str((int(r) + noise)%period)
        elif corrput_location == 'h':
            h = str((int(h) + noise)%period)
        else:
            raise Exception('No such corruption location %s'%corrput_location)

        res.append([h, r, t])
    return np.array(res)

def traj_graph(lis1, lis2, name='', viz= False, lis1_only = False):
    # lis = range(2, 20)
    n = len(lis1)
    assert len(lis1) == len(lis2)
    edges = [(lis1[0], lis1[1])]
    for i in range(1, n-1):
        edge1 = (lis1[i], lis1[i + 1])
        edges.append(edge1)
        if not lis1_only:
            edge2 = (lis1[i], lis2[i + 1])
            edges.append(edge2)

    g = nx.Graph()
    g.add_edges_from(edges)
    pos = nx.spring_layout(g)
    # pos = nx.circular_layout(g)
    nx.draw(g, pos, node_color='b', node_size=5, with_labels=True, width=1)
    try:
        os.makedirs('./img')
    except FileExistsError:
        pass
    if viz: plt.show()
    plt.savefig('./img/traj_graph' + name)

# lis1 = [2, 3, 4, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 35, 36, 37, 38, 39, 40, 41, 42]
# lis2 = [34, 74, 35, 89, 30, 14, 37, 15, 32, 40, 61, 0, 1, 16, 61, 25, 95, 88, 30, 31, 29, 79, 58, 68, 41, 64, 85, 36, 70, 68, 30, 14, 37, 15, 32, 40, 61, 0, 1, 16, 61, 25, 95, 88, 30, 31, 29, 79, 58, 68, 41, 64, 85, 36, 70, 68, 30, 14, 37, 15, 32, 40, 61, 0, 1, 16, 61, 25, 95, 88, 30, 31, 29, 79, 58, 68, 41, 64, 85, 36, 70, 68, 30, 14, 37, 15, 32, 40, 61, 0, 1, 16, 61, 25, 95, 88, 30, 31, 29, 79]
# traj_graph(lis1, lis2, name='', viz=True, lis1_only=True)
# sys.exit()

@ex.config
def get_config():
    # unused params
    depth = 8
    rb = True

    # param
    verbose = True
    data = 'single_fam_tree'
    seed = 42
    noise_rel = 8
    n_node = 1000
    model = 'ComplEx'
    rels = '1 2 3 4 5 6 7 8 9 10 11 12'#'1 2 3 4 5 6 7 8 9 10 11 12' #'1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16'

    # other param
    period = 3

@ex.main
def main(model, n_node, noise_rel, seed, data, verbose, rels, period):
    sys.argv = []
    args = parser.parse_args()
    args.model = model
    args.n_node = n_node
    args.noise_rel = noise_rel
    args.seed = seed
    args.data = data
    args.verbose = verbose
    args.rels = rels
    args.period = period
    print(args)

    if args.data == 'single_fam_tree':
        py_server = ['/home/cai.507/anaconda3/envs/ampligraph/bin/python']
        py_local = ['~/anaconda3/bin/python']
        tmp_command = ['../ampligraph/datasets/single_fam_tree.py',
                                       '--seed', str(args.seed), '--add', str(args.add),
                                       '--extra_rel', str(args.extra_rel),
                                       '--noise_rel', str(args.noise_rel),
                                       '--rels', str(args.rels),
                                       '--n_node', str(args.n_node)]
        command_server = py_server + tmp_command
        command_local = py_local + tmp_command
    else:
        raise ('No such data %s'%args.data)

    try:
        print(command_server)
        print(subprocess.check_output(command_server))
    except FileNotFoundError:
        print(command_local)
        print(subprocess.check_output(command_local))

    X, _ = load_data(args)
    for key, val in X.items():
        print(key, len(val),)

    # get model
    model = get_model(args)

    # Fit the model on training  and validation set
    filter = np.concatenate((X['train'], X['valid'], X['test']))
    print(model,'\n')
    model.fit(X['train'], early_stopping=True,
              early_stopping_params= \
                  {   'x_valid': X['valid'],  # validation set
                      'criteria': 'hits10',  # Uses hits10 criteria for early stopping
                      'burn_in': 100,  # early stopping kicks in after 100 epochs
                      'check_interval': 20,  # validates every 20th epoch
                      'stop_interval': 5,  # stops if 5 successive validation checks are bad.
                      'x_filter': filter,  # Use filter for filtering out positives
                      'corruption_entities': 'all',  # corrupt using all entities
                      'corrupt_side': 's+o'  # corrupt subject and object (but not at once)
                  })

    # Run the evaluation procedure on the test set (with filtering). Usually, we corrupt subject and object sides separately and compute ranks
    # ranks = evaluate_performance(X['test'], model=model, filter_triples=filter, verbose=args.verbose, use_default_protocol=True)  # corrupt subj and obj separately while evaluating
    eval_corrution(model, filter, args, X['test'], noise=0, noise_loc='t')
    eval_corrution(model, filter, args, X['test'], noise=1, noise_loc='t')
    # eval_corrution(model, filter, args, X['test'], noise=1, noise_loc='h')

    queries = []
    best_tails, second_best_tails = [], []
    step = 2

    edges = []
    for i in range(n_node):
        query = [str(i), str(step)]
        best_tail, second_tail = topk_tails(model, hr = np.array(query), top_k=3, print_f=True)
        edge = (i, second_tail)
        edges.append(edge)
    print(edges)
    g = nx.Graph()
    g.add_edges_from(edges)
    # pos = nx.spring_layout(g, seed=42)
    pos = nx.circular_layout(g)
    nx.draw(g, pos, node_color='b', node_size=5, with_labels=True, width=1)
    name = './img/circular_layout_second/rel_' + str(step) + '_' + str(n_node) + '_' + str(rels) # TODO change circular layout
    title = f'{n_node} nodes. Rels {rels}. relation step {step}'
    plt.title(title)
    plt.savefig(name)
    sys.exit()

    for i in range(n_step):
        query = [str(h), str(step)]
        print('iter %s: head %s'%(i, query[0]))
        queries.append(query)
        best_tail, second_best_tail = topk_tails(model, hr=np.array(query), top_k=5, print_f=False)
        best_tails.append(best_tail)
        second_best_tails.append(second_best_tail)
        h = best_tail

    print(best_tails)
    print(second_best_tails)
    traj_graph(best_tails, second_best_tails, name='_' + str(n_step))




if __name__ == "__main__":
    # main('ComplEx', 1000, 1, 42, 'single_fam_tree', True)
    ex.run_commandline()  # SACRED: this allows you to run Sacred not only from your terminal,
