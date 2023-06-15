# Graph-ml
| Books and Resources | Status of Completion |
| ----- | -----|
| 1. **Hands-On Graph Neural Networks Using Python** | |
| 2. **Graph-Powered Machine Learning** | |
| 3. **Graph Machine Learning** | |



Why graph learning?
Graph learning is the application of machine learning techniques to graph data. This study area encompasses a range of tasks aimed at understanding and manipulating graph-structured data. There are many graphs learning tasks, including the following:

- Node classification is a task that involves predicting the category (class) of a node in a graph. For example, it can categorize online users or items based on their characteristics. In this task, the model is trained on a set of labeled nodes and their attributes, and it uses this information to predict the class of unlabeled nodes.
- Link prediction is a task that involves predicting missing links between pairs of nodes in a graph. This is useful in knowledge graph completion, where the goal is to complete a graph of entities and their relationships. For example, it can be used to predict the relationships between people based on their social network connections (friend recommendation).
- Graph classification is a task that involves categorizing different graphs into predefined categories. One example of this is in molecular biology, where molecular structures can be represented as graphs, and the goal is to predict their properties for drug design. In this task, the model is trained on a set of labeled graphs and their attributes, and it uses this information to categorize unseen graphs.
- Graph generation is a task that involves generating new graphs based on a set of desired properties. One of the main applications is generating novel molecular structures for drug discovery. This is achieved by training a model on a set of existing molecular structures and then using it to generate new, unseen structures. The generated structures can be evaluated for their potential as drug candidates and further studied.
- Graph learning has many other practical applications that can have a significant impact. One of the most well-known applications is recommender systems, where graph learning algorithms recommend relevant items to users based on their previous interactions and relationships with other items. Another important application is traffic forecasting, where graph learning can improve travel time predictions by considering the complex relationships between different routes and modes of transportation.

The versatility and potential of graph learning make it an exciting field of research and development. The study of graphs has advanced rapidly in recent years, driven by the availability of large datasets, powerful computing resources, and advancements in machine learning and artificial intelligence. As a result, we can list four prominent families of graph learning techniques [1]:

- Graph signal processing, which applies traditional signal processing methods to graphs, such as the graph Fourier transform and spectral analysis. These techniques reveal the intrinsic properties of the graph, such as its connectivity and structure.
- Matrix factorization, which seeks to find low-dimensional representations of large matrices. The goal of matrix factorization is to identify latent factors or patterns that explain the observed relationships in the original matrix. This approach can provide a compact and interpretable representation of the data.
- Random walk, which refers to a mathematical concept used to model the movement of entities in a graph. By simulating random walks over a graph, information about the relationships between nodes can be gathered. This is why they are often used to generate training data for machine learning models.
- Deep learning, which is a subfield of machine learning that focuses on neural networks with multiple layers. Deep learning methods can effectively encode and represent graph data as vectors. These vectors can then be used in various tasks with remarkable performance.

![11](https://github.com/andysingal/Graph-ml/blob/main/resources/Screenshot%202023-06-13%20at%207.39.25%20AM.png)

Graph theory is a fundamental branch of mathematics that deals with the study of graphs and networks. A graph is a visual representation of complex data structures that helps us understand the relationships between different entities. Graph theory provides us with tools to model and analyze a vast array of real-world problems, such as transportation systems, social networks, and internet connectivity.

It cover the following main topics:

- Introducing graph properties
- Discovering graph concepts
- Exploring graph algorithms

<h3>Weighted graphs</h3>
Another important property of graphs is whether the edges are weighted or unweighted. In a weighted graph, each edge has a weight or cost associated with it. These weights can represent various factors, such as distance, travel time, or cost.

For example, in a transportation network, the weights of edges might represent the distances between different cities or the time it takes to travel between them. In contrast, unweighted graphs have no weight associated with their edges. These types of graphs are commonly used in situations where the relationships between nodes are binary, and the edges simply indicate the presence or absence of a connection between them.

<h2>Types of graphs</h2>

In addition to the commonly used graph types, there are some special types of graphs that have unique properties and characteristics:

- A tree is a connected, undirected graph with no cycles. Since there is only one path between any two nodes in a tree, a tree is a special case of a graph. Trees are often used to model hierarchical structures, such as family trees, organizational structures, or classification trees.
- A rooted tree is a tree in which one node is designated as the root, and all other vertices are connected to it by a unique path. Rooted trees are often used in computer science to represent hierarchical data structures, such as filesystems or the structure of XML documents.
- A directed acyclic graph (DAG) is a directed graph that has no cycles. This means that the edges can only be traversed in a particular direction, and there are no loops or cycles. DAGs are often used to model dependencies between tasks or events – for example, in project management or in computing the critical path of a job.
- A bipartite graph is a graph in which the vertices can be divided into two disjoint sets, such that all edges connect vertices in different sets. Bipartite graphs are often used in mathematics and computer science to model relationships between two different types of objects, such as buyers and sellers, or employees and projects.
- A complete graph is a graph in which every pair of vertices is connected by an edge. Complete graphs are often used in combinatorics to model problems involving all possible pairwise connections, and in computer networks to model fully connected networks.

![21](https://github.com/andysingal/Graph-ml/blob/main/resources/Screenshot%202023-06-14%20at%208.10.55%20AM.png)

<h3>Discovering graph concepts</h3>
<h4>Graph measures</h4>
Centrality quantifies the importance of a vertex or node in a network. It helps us to identify key nodes in a graph based on their connectivity and influence on the flow of information or interactions within the network. There are several measures of centrality, each providing a different perspective on the importance of a node:

Degree centrality is one of the simplest and most commonly used measures of centrality. It is simply defined as the degree of the node. A high degree centrality indicates that a vertex is highly connected to other vertices in the graph, and thus significantly influences the network.
Closeness centrality measures how close a node is to all other nodes in the graph. It corresponds to the average length of the shortest path between the target node and all other nodes in the graph. A node with high closeness centrality can quickly reach all other vertices in the network.
Betweenness centrality measures the number of times a node lies on the shortest path between pairs of other nodes in the graph. A node with high betweenness centrality acts as a bottleneck or bridge between different parts of the graph.

<h3>Exploring graph algorithms</h3>
Graph algorithms are critical in solving problems related to graphs, such as finding the shortest path between two nodes or detecting cycles. This section will discuss two graph traversal algorithms: BFS and DFS.

<h4>Breadth-first search</h4>
BFS is a graph traversal algorithm that starts at the root node and explores all the neighboring nodes at a particular level before moving to the next level of nodes. It works by maintaining a queue of nodes to visit and marking each visited node as it is added to the queue. The algorithm then dequeues the next node in the queue and explores all its neighbors, adding them to the queue if they haven’t been visited yet.

![23](https://github.com/andysingal/Graph-ml/blob/main/resources/Screenshot%202023-06-14%20at%208.20.40%20AM.png)

![24](https://github.com/andysingal/Graph-ml/blob/main/resources/Screenshot%202023-06-14%20at%208.22.43%20AM.png)

DeepWalk: 
DeepWalk is one of the first major successful applications of machine learning (ML) techniques to graph data. It introduces important concepts such as embeddings that are at the core of GNNs. Unlike traditional neural networks, the goal of this architecture is to produce representations that are then fed to other models, which perform downstream tasks (for example, node classification).

In this chapter, we will learn about the DeepWalk architecture and its two major components: Word2Vec and random walks. We’ll explain how the Word2Vec architecture works, with a particular focus on the skip-gram model. We will implement this model with the popular gensim library on a natural language processing (NLP) example to understand how it is supposed to be used.

Then, we will focus on the DeepWalk algorithm and see how performance can be improved using hierarchical softmax (H-Softmax). This powerful optimization of the softmax function can be found in many fields: it is incredibly useful when you have a lot of possible classes in your classification task. We will also implement random walks on a graph before wrapping things up with an end-to-end supervised classification exercise on Zachary’s Karate Club.

By the end of this chapter, you will master Word2Vec in the context of NLP and beyond. You will be able to create node embeddings using the topological information of the graphs and solve classification tasks on graph data.

In this chapter, we will cover the following main topics:

- Introducing Word2Vec
- DeepWalk and random walks
-  Implementing DeepWalk

![22](https://github.com/andysingal/Graph-ml/blob/main/resources/Screenshot%202023-06-15%20at%208.29.27%20AM.png)

![23](https://github.com/andysingal/Graph-ml/blob/main/resources/Screenshot%202023-06-15%20at%208.31.52%20AM.png)
