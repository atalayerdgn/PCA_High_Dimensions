import numpy as np

def normalize(X):

    N, D = X.shape
    mu = np.zeros((D,))
    mu = np.mean(X,axis = 0)
    Xbar = X - mu
    return Xbar, mu
def eig(S):

    eigvals, eigvecs = np.linalg.eig(S)
    sort_indices = np.argsort(eigvals)[::-1]
    return eigvals[sort_indices], eigvecs[:, sort_indices]

def projection_matrix(B):
    P = B @ np.linalg.inv(B.T @ B) @ B.T 
    return np.eye(B.shape[0]) @ P

def PCA(X, num_components):

    X_normalized, mean = normalize(X)
    S = np.cov(X_normalized, rowvar=False, bias=True)
    eig_vals, eig_vecs = eig(S)
    principal_vals, principal_components = eig_vals[:num_components] ,eig_vecs[:, :num_components]
    principal_components = np.real(principal_components) 
    P = projection_matrix(eig_vecs[:,:num_components])
    x_reconst = P@X_normalized.T
    reconst = x_reconst.T + mean
    return reconst, mean, principal_vals, principal_components

def PCA_high_dim(X, num_components):
    N, D = X.shape
    X_normalized, mean = normalize(X)
    M = np.dot(X_normalized, X_normalized.T) / N
    eig_vals, eig_vecs = eig(M)
    eig_vecs = X_normalized.T @ eig_vecs
    principal_values = eig_vals[:num_components]
    principal_components = eig_vecs[:, :num_components]
    principal_components = np.real(principal_components)
    print(projection_matrix(principal_components).shape)
    reconst = (projection_matrix(principal_components) @ X_normalized.T).T + mean
    return reconst, mean, principal_values, principal_components

random = np.random.RandomState(0)
X = random.randn(5, 4)

print(PCA(X, 2))
print(PCA_high_dim(X, 2))
