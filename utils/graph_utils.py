def calculate_graph_metrics(graph):
    # Function to calculate various metrics for a given graph
    metrics = {}
    metrics["num_nodes"] = len(graph.nodes)
    metrics["num_edges"] = len(graph.edges)
    metrics["density"] = calculate_density(graph)
    metrics["average_degree"] = calculate_average_degree(graph)
    return metrics


def calculate_density(graph):
    # Function to calculate the density of the graph
    if len(graph.nodes) == 0:
        return 0
    return len(graph.edges) / (len(graph.nodes) * (len(graph.nodes) - 1))


def calculate_average_degree(graph):
    # Function to calculate the average degree of the graph
    if len(graph.nodes) == 0:
        return 0
    return 2 * len(graph.edges) / len(graph.nodes)


def visualize_graph(graph):
    # Function to visualize the graph using a plotting library
    import matplotlib.pyplot as plt
    import networkx as nx

    plt.figure(figsize=(10, 7))
    nx.draw(graph, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.title("Graph Visualization")
    plt.show()
