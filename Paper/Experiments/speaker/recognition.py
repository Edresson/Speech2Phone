import numpy as np
try:
    import _pickle as pickle #Python3 support
except:
    import cPickle as pickle#Python2 support
    
try:
    from speaker import GMMSet, GMM #Gpu Support
    pass
except:
    print("pycaspgmm")
    import operator
    import numpy as np
    from sklearn.mixture import GaussianMixture as GMM
    class GMMSet(object):

        def __init__(self, gmm_order = 32):
            self.gmms = []
            self.gmm_order = gmm_order
            self.y = []

        def fit_new(self, x, label):
            self.y.append(label)
            gmm = GMM()
            gmm.fit(x)
            self.gmms.append(gmm)
            
        def fit_multi_instances_same_class(self, x, label):
            
            if label in self.y:
                idx = self.y.index(label)
                gmm = self.gmms[idx]
                gmm.fit(x)
            else:    
                self.y.append(label)
                gmm = GMM(n_components=13)
                gmm.fit(x)
                self.gmms.append(gmm)
        def gmm_score(self, gmm, x):
            return np.sum(gmm.score(x))

        def predict_one(self, x):
            scores = [self.gmm_score(gmm, x) / len(x) for gmm in self.gmms]
            p = sorted(enumerate(scores), key=operator.itemgetter(1), reverse=True)
            result = [(self.y[index], value) for (index, value) in enumerate(scores)]
            p = max(result, key=operator.itemgetter(1))
            return p
    
    




    
class GMMRec(object):

    def __init__(self):
        self.features = []
        self.gmmset = GMMSet()
        self.classes = []
        
    def enroll(self, name, mfcc_vecs):
        mu = np.mean(mfcc_vecs, axis = 0)
        sigma = np.std(mfcc_vecs, axis = 0)
        feature = (mfcc_vecs - mu) / sigma
        feature = feature.astype(np.float32)
        self.features.append(feature)
        self.classes.append(name)

    def _get_gmm_set(self):
        return GMMSet()

    def train(self):
        self.gmmset = self._get_gmm_set()
        for name, feats in zip(self.classes, self.features):
            self.gmmset.fit_new(feats, name)
            
    def predict(self, mfcc_vecs):
        mu = np.mean(mfcc_vecs, axis = 0)
        sigma = np.std(mfcc_vecs, axis = 0)
        feature = (mfcc_vecs - mu) / sigma
        feature = feature.astype(np.float32)
        return self.gmmset.predict_one(feature)
            
    def dump(self, fname, part = None):
        try:
            with open(fname, 'wb') as f:
                if part is None:
                    pickle.dump(self, f, -1)
                else:
                    pickle.dump(part, f, -1)
        except:
            with open(fname, 'w') as f:
                if part is None:
                    pickle.dump(self, f, -1)
                else:
                    pickle.dump(part, f, -1)

    @staticmethod
    def load(fname):
        try:
            
            with open(fname, 'rb') as f:
                R = pickle.load(f)
                return R
        except:
            with open(fname, 'r') as f:
                R = pickle.load(f)
                return R

            
