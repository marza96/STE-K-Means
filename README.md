# STE-K-Means

This repository contains an implementation of a differentiable K-Means solver using the Straight Through Estimator. 


## Implementation
The core principle relies on considering the K-Means as a matrix factorization problem. Specifically the data points $\mathbf{x}_i$ are concatenated in a data matrix $\mathbf{X}$ given as 
$$
    \mathbf{X} = \begin{bmatrix} \mathbf{x}_1^T \\
        \mathbf{x}_2^T \\
        \vdots \\
        \mathbf{x}_n^T \end{bmatrix}
$$
Then the result from [Baukhage 2023] is used to formulate the K-Means problem as a low-rank matrix factorization problem of the form:
$$
    \min_{\mathbf{U}} \|\mathbf{X} - \mathbf{U}(\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T \|_F^2 \\
$$
such that:
$$
    \mathbf{U} \in \{0,1\}^{k \times n}
$$
and:
$$
    \sum_i \mathbf{U} = 1.
$$
In the STE context this problem can be solved by imposing defining trainable weight matrices $\mathbf{D} = \text{Diag}(1, \dots, 1) \in \mathbb{R}^{k \times k}$ and $\mathbf{W} \in \mathbb{R}^{n - k \times n}$ such that $\mathbf{U}$ is evaluated as:
$$
    \mathbf{U}^T = \begin{bmatrix}\mathbf{D} & \mathbf{g}(\mathbf{W}) \end{bmatrix} 
$$  
where $\mathbf{g}(\cdot)$ is te Gumbel-STE operator. The optimization problem is then optimized using Adam or SGD. 

### Kmeans
![Model Folding Concept Figure](figures/km.png)

### STE Kmeans
![Model Folding Concept Figure](figures/km.png)