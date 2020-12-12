import jraph
import jax.numpy as jnp

if __name__ == '__main__':

    # define a three node graph, each node has integer as feature.
    node_features = jnp.array([[0.], [1.], [2.]])

    # construct a graph from which there is a directed edge between each node and its successor
    # define senders (source nodes) and 'recievers' (destination nodes)
    # (destination nodes)

    senders = jnp.array([0, 1, 2])
    recievers = jnp.array([1, 2, 0])

    # add edge attributes
    edges = jnp.array([[5.], [6.], [7.]])

    # We can save number of nodes and edges
    # Used to make running GNNs over multiple graphs in a GraphsTuple possible
    n_nodes = jnp.array([3])
    n_edges = jnp.array([3])

    # add global information
    global_context = jnp.array([[1]])
    graph = jraph.GraphsTuple(nodes=node_features, senders=senders, recievers=recievers,
        edges=edges, n_nodes=n_nodes, n_edges=n_edges, globals=global_context)
    
    two_graph_graphstuple = jraph.batch([graph, graph])
    
    # Tell which nodes are from which graph using n_node
    jraph.batch([graph, graph]).nodes
    jraph.batch([graph, graph]).n_node

    # node targets
    node_targets = jnp.array([[True], [False], [True]])
    graph = graph._replace(nodes={'inputs':graph.nodes, 'targets': node_targets})


    ### Using the model zoo

    # update graph features
    def update_edge_fn(edge, sender, reciever, globals_):
        return edge

    def update_node_fn(node, sender, reciever, globals_):
        return node

    def update_global_fn(node, sender, reciever, globals_):
        return globals_

    @jraph.concatenaged_args
    def update_edge_fn(concatenated_features):
        return concatenated_features
    
    net = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn
    )

    update_graph = net(graph)

