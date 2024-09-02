import numpy as np
from sklearn.svm import SVC


class ADASVM_TW():
    def __init__(self, t):
        self.models = None
        self.vws = None
        self.t = t
    def fit(self, X, y, z = 0.42, c = 0.05, U = 5):
        models = []
        vws = []
        m = len(X)
        W = np.full(m, 1/m)
        #print(W[y==1])
        #print()
        #P = D[D['h3'] == 1].index
        #N = D[D['h3'] == 0].index
        u_t = 0
        while u_t < U:
            #print(u_t)
            subset = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True, p=W)
            X_u = X[subset]
            y_u = y[subset]
            svm = SVC(probability = True)
            svm.fit(X_u,y_u)      
            y_pred = svm.predict(X)
            e = (y != y_pred).astype(int)
            e_u = np.dot(W, e) / np.sum(W)
            if e_u <= c or e_u >= 0.5:
                continue
            else:
                models.append(svm)
                l_u = np.where(y == y_pred, 1, -1)
                a_u = 0.5 * np.log((1 - e_u) / e_u)
                #print(z * t * l_u)
                vws.append(a_u)
                tl = np.exp(-a_u * (l_u * np.exp(z * self.t * l_u)))
                #print(tl)
                W = W * tl
                W = W / np.sum(W)
                #print(np.sum(W))
                u_t += 1
        self.models = models
        self.vws = vws
    def predict(self,X_test):
        preds = np.array([m.predict(X_test) for m in self.models])
        w_sum = np.dot(self.vws, preds)
        final_preds = np.sign(w_sum).astype(int)
        return final_preds
    def predict_proba(self,X_test):
        n_samples = X_test.shape[0]
        n_classes = 2
        prob = np.zeros((n_samples, n_classes))
        
        # Iterate over each model
        for model, vw in zip(self.models, self.vws):
            # Get probabilities from the model
            model_prob = model.predict_proba(X_test)
            # Accumulate weighted probabilities
            prob += vw * model_prob
        
        # Normalize probabilities so that they sum to 1
        prob_sum = np.sum(prob, axis=1, keepdims=True)
        prob /= prob_sum
        
        return prob