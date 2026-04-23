from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def run_kmeans(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels

def run_isolation_forest(X):
    model = IsolationForest(random_state=42)
    labels = model.fit_predict(X)
    return labels

def train_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def train_random_forest(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def train_svm(X, y):
    model = SVC()
    model.fit(X, y)
    return model
