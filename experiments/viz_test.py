# from importlib import reload  # Python 3.4+ only.
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def viz_graph(g, node_size = 5, edge_width = 1, node_color = 'b', color_bar = False, show = False):
    # g = nx.random_geometric_graph(100, 0.125)
    pos = nx.spring_layout(g)
    nx.draw(g, pos, node_color=node_color, node_size=node_size, with_labels=False, width = edge_width)
    if color_bar:
        # https://stackoverflow.com/questions/26739248/how-to-add-a-simple-colorbar-to-a-network-graph-plot-in-python
        sm = plt.cm.ScalarMappable( norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
        sm._A = []
        plt.colorbar(sm)
    if show: plt.show()

def test():
    G = nx.star_graph(20)
    pos = nx.spring_layout(G)
    colors = range(20)
    cmap = plt.cm.Blues
    vmin = min(colors)
    vmax = max(colors)
    nx.draw(G, pos, node_color='#A0CBE2', edge_color=colors, width=4, edge_cmap=cmap,
            with_labels=False, vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)
    plt.show()

def sample():
    G = nx.random_geometric_graph(200, 0.125)
    pos = nx.get_node_attributes(G, 'pos')

    # find node near center (0.5,0.5)
    dmin = 1
    ncenter = 0
    for n in pos:
        x, y = pos[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d

    # color by path length from node near center
    p = dict(nx.single_source_shortest_path_length(G, ncenter))

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=list(p.keys()),
                           node_size=8,
                           node_color=list(p.values()),
                           cmap=plt.cm.Reds_r)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis('off')
    plt.show()

def viz_deghis(G):
    # G = nx.gnp_random_graph(100, 0.02)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()

def multi_plots(n_row = 3, n_col = 3):
    # First create some toy data:
    x = np.linspace(0, 2*np.pi, 400)
    y = np.sin(x**2)

    fig, axes = plt.subplots(n_row, n_col)
    for row_idx in range(n_row):
        for col_idx in range(n_col):
            axes[row_idx, col_idx].plot(x,y)
            axes[row_idx, col_idx].set_title(str(col_idx) + ' ' + str(row_idx))

    plt.show()


if __name__=='__main__':
    multi_plots()
