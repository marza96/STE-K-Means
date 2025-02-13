import torch
import numpy as np

from tqdm import tqdm


def kmeans_plus_plus_init(X, k, return_labels=False):
    n_samples, _ = X.shape
    centers = np.empty((k, X.shape[1]))
    
    centers[0] = X[np.random.randint(n_samples)]
    
    for i in range(1, k):
        dists = np.min(np.linalg.norm(X[:, np.newaxis] - centers[:i], axis=2) ** 2, axis=1)
        
        probabilities = dists / np.sum(dists)
        centers[i] = X[np.random.choice(n_samples, p=probabilities)]
    
    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(distances, axis=1)
    
    if return_labels:
        return labels
    else:
        clustering_matrix = np.zeros((n_samples, k))
        clustering_matrix[np.arange(n_samples), labels] = 1
        return clustering_matrix.T


class STEClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def top_k_mask(self, M, k):
        max_indices = torch.argmax(M, dim=0)
        N = torch.zeros_like(M)
        N[max_indices, torch.arange(M.shape[1])] = 1

        return N

    def random_sparse_matrix(self, m, n, k):
        matrix = torch.zeros((m, n))
        row_indices = torch.randint(0, m, (n,))
        matrix[row_indices, torch.arange(n)] = torch.randn(n) 
        m = (matrix != 0).float()

        return m
    
    def fit_predict(self, X, epochs=6500, lr=0.05):
        X = torch.tensor(X).float().to("cuda")
        C = torch.tensor(kmeans_plus_plus_init(X.cpu().numpy(), self.n_clusters, return_labels=False)).cuda().float()
        D = C[:, :self.n_clusters].detach().clone()
        R = C[:, self.n_clusters:].detach().clone()

        D.requires_grad = False
        R.requires_grad = True

        best_loss = 1e25
        best_C    = torch.hstack(
            (
                D, 
                self.top_k_mask(R, X.shape[0] - self.n_clusters)
            )
        ).detach()

        optimizer = torch.optim.Adam([R], lr=lr)

        with tqdm(range(epochs), desc="Solving") as pbar:
            for epoch in pbar:
                optimizer.zero_grad()   

                M = torch.hstack((D.detach(), torch.nn.functional.gumbel_softmax(R, tau=1, hard=True, dim=0)))
                loss = torch.nn.functional.mse_loss(X, M.T @ (torch.diag(1.0 / torch.diag(M @ M.T))) @ M @ X)

                if loss < best_loss:
                    best_loss = loss
                    best_C = torch.hstack((D, self.top_k_mask(R, X.shape[0] - self.n_clusters))).detach()
                    pbar.set_postfix(loss=f"{loss:.4f}")

                loss.backward() 
                optimizer.step() 

        C = best_C

        labels = np.ones(C.shape[1])
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if C[i, j] == 1:
                    labels[j] = i

        return labels