import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def E_SMOTE(X, W, n_generate, k=3):
    """
    Perform SMOTE oversampling.

    Parameters:
    - X: Feature matrix - numpy array, shape (n_samples, n_features)
    - W: Weights array of each samples - numpy array, shape (n_samples, 1)
    - n_generate: Number of synthetic samples needed to generate for each real datapoints - 
    numpy array, shape (n_samples, 1)

    Returns:
    - synthetic_samples: Generated synthetic data - numpy array, shape (n_generate_total, n_features)
    """

    # Use k-nearest neighbors to generate synthetic samples
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)

    # Generate synthetic samples
    synthetic_samples = []
    synthetic_weights = []
    i = 0
    while i < X.shape[0]:
        n_samples = n_generate[i]
        while n_samples > 0:
            x_min = X[i]
            # Find its k-nearest neighbors
            neighbors = knn.kneighbors([x_min], return_distance=False)
            nn = neighbors[0][np.random.randint(1, k)]  # Exclude the sample itself
    
            # Generate a synthetic sample
            gap = np.random.random()
            synthetic_sample = x_min + gap * (X[nn] - x_min)
            synthetic_samples.append(synthetic_sample)
            synthetic_weights.append(W[i])
            n_samples -= 1
        i += 1

    synthetic_samples = np.array(synthetic_samples)

    return synthetic_samples



class E_SMOTE_ADASVM_TW():
    """
    E_SMOTE_ADASVM_TW class implements an ensemble model using SMOTE 
    and AdaBoost with SVM as the base classifier.
    Parameters:
    - models: Base models.
    - vws: The voting weights for each base models.
    - t: Time of the data used for time weighting.
    """
    
    def __init__(self, t):
        self.models = None
        self.vws = None
        self.t = t

    def calculate_synthetic(self, X, y, W):
        """
        Calculate the number of synthetic samples to generate.
        
        Returns None if no synthetic samples are needed.
        """
        counter = Counter(y)
        SNP = counter[-1] - counter[1]  # Calculate the imbalance
        if SNP <= 0:
            return None
        
        Nu = np.round(W * SNP)  # Initial estimate of synthetic samples
        
        # Adjust the number of synthetic samples to exactly match SNP
        if SNP > np.sum(Nu):
            Mu = SNP - np.sum(Nu)
            js = np.argsort(-W)[:int(Mu)]
            Nu[js] += 1
        else:
            Mu = SNP - np.sum(Nu)
            js = np.argsort(W)[:int(Mu)]
            Nu[js] -= 1
        
        return Nu
        
    def fit(self, X, y, z=0.42, c=0.05, U=5, balanced=True):
        """
        Train the ensemble model using AdaBoost with SVM classifiers.
        
        Args:
            z: Parameter for updating weights.
            c: Error threshold for stopping criteria.
            U: Number of base classifiers to train.
            balanced: Whether to balance the dataset using SMOTE.
        """
        t = self.t
        models = []
        vws = []
        m = len(X)
        W = np.full(m, 1/m)  # Initialize weights uniformly

        u_t = 0
        while u_t < U:
            # Separate positive and negative samples
            XP = X[y == 1]
            XN = X[y == -1]
            
            WP = W[y == 1] / np.sum(W[y == 1])
            WN = W[y == -1] / np.sum(W[y == -1])
            N_subset = np.random.choice(np.arange(XN.shape[0]), size=XN.shape[0], replace=True, p=WN)
            XN = XN[N_subset]
            
            if not balanced:
                # Generate synthetic samples if the dataset is imbalanced
                N_generate = self.calculate_synthetic(XP, y, WP)
                if N_generate is not None:
                    X_SMOTE_P = E_SMOTE(XP, WP, N_generate)
                    X_u = np.vstack((XP, X_SMOTE_P, XN))
                    y_u = np.hstack((np.full(XP.shape[0] + X_SMOTE_P.shape[0], 1), np.full(XN.shape[0], -1)))
                else:
                    X_u = X
                    y_u = y
            else:
                X_u = X
                y_u = y
            
            svm = SVC(probability=True)
            svm.fit(X_u, y_u)
            y_pred = svm.predict(X)
            e = (y != y_pred).astype(int)  # Error vector
            e_u = np.dot(W, e) / np.sum(W)  # Weighted error rate
            
            if e_u <= c or e_u >= 0.5:
                continue  # Skip this model if error is too low or too high
            
            models.append(svm)
            l_u = np.where(y == y_pred, 1, -1)
            a_u = 0.5 * np.log((1 - e_u) / e_u)  # Calculate model weight
            vws.append(a_u)
            tl = np.exp(-a_u * (l_u * np.exp(z * t * l_u)))  # Update weights
            W = W * tl
            W = W / np.sum(W)  # Normalize weights
            u_t += 1
        
        self.models = models
        self.vws = vws

    def predict(self, X_test):
        preds = np.array([m.predict(X_test) for m in self.models])
        w_sum = np.dot(self.vws, preds)
        final_preds = np.sign(w_sum).astype(int)  # Aggregate predictions
        return final_preds

    def predict_proba(self, X_test):
        n_samples = X_test.shape[0]
        n_classes = 2
        prob = np.zeros((n_samples, n_classes))
        
        # Accumulate weighted probabilities from each model
        for model, vw in zip(self.models, self.vws):
            model_prob = model.predict_proba(X_test)
            prob += vw * model_prob
        
        prob_sum = np.sum(prob, axis=1, keepdims=True)
        prob /= prob_sum  # Normalize probabilities
        
        return prob