import networkx as nx
import matplotlib.pyplot as plt

edges = set()


def visualize(matrix, docs):
    for i, row in enumerate(matrix):
        for j, score in enumerate(row):
            pair = [str(docs[i]), str(docs[j])]
            pair.sort()
            if str(docs[i]) != str(docs[j]):
                edges.add((pair[0], pair[1], score))

    G = nx.Graph()
    NG = nx.Graph()

    for doc in docs:
        G.add_node(doc)
        #NG.add_node(doc)

    for edge in edges:
        print(edge[0], edge[1], edge[2])
        G.add_edge(edge[0], edge[1], weight=edge[2])

    for thing in G.edges:
        print(G.get_edge_data(thing[0], thing[1]))
        if G.get_edge_data(thing[0], thing[1])['weight'] > .05:
            NG.add_node(thing[0])
            NG.add_node(thing[1])
            NG.add_edge(thing[0], thing[1], weight=G.get_edge_data(thing[0], thing[1])['weight'])

    nx.draw(NG, pos=nx.spring_layout(G), with_labels=True, font_size=6)

    plt.figure(figsize=(60, 60), dpi=600)
    plt.show()

def visualize2(matrix, docs):
    G = nx.from_numpy_matrix(matrix)
    #G = nx.relabel_nodes(G, lambda x: docs[x])
    G.edges(data=True)


    edges = []
    weights = []

    for i, j in nx.get_edge_attributes(G, 'weight').items():
        if j > .999:
            edges.append(i)
            weights.append(j)
    #edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    # positions
    positions = nx.circular_layout(G)

    # Figure size
    plt.figure(figsize=(15, 15))

    # draws nodes
    nx.draw_networkx_nodes(G, positions, node_color='#DA70D6',
                           node_size=500, alpha=0.8)

    # Styling for labels
    nx.draw_networkx_labels(G, positions, font_size=8,
                            font_family='sans-serif')

    # draws the edges
    nx.draw_networkx_edges(G, positions, edge_list=edges, style='solid')

    # displays the graph without axis
    plt.axis('off')
    # saves image
    plt.savefig("part1.png", format="PNG")
    plt.show()
