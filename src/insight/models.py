import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.algorithm = None
        self.pipeline = None
        self.model_kws = cfg["model_kw"]
        self.best_k = None
        self.preprocessor = None

        # Set algorithm based on configuration
        if cfg["alg"] == "KMeans":
            if self.model_kws.get('n_clusters') == -1:
                self.algorithm = None  # Will be set after silhouette analysis
            else:
                self.algorithm = KMeans(**self.model_kws)
        elif cfg["alg"] == "DBSCAN":
            self.algorithm = DBSCAN(**self.model_kws)
        elif cfg["alg"] == "GMM":
            self.algorithm = GaussianMixture(**self.model_kws, covariance_type="tied", random_state=42)
        else:
            raise NotImplementedError(f"Algorithm {cfg['alg']} is not implemented")

    def _create_preprocessor(self, X,):
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

        categorical_transformer = Pipeline([
            ("cat_imp", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore"))
        ])

        numerical_transformer = Pipeline([
                ("num_imp", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
        ])

        self.preprocessor = ColumnTransformer([
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ])
        return self.preprocessor


    def perform_silhouette_analysis(self, X):
        """Perform Silhouette Analysis to find the optimal number of clusters."""
        # Preprocess the data
        preprocessor = self._create_preprocessor(X)
        X_preprocessed = preprocessor.fit_transform(X)
        X_dense = X_preprocessed.toarray() if hasattr(X_preprocessed, 'toarray') else X_preprocessed
        
        # Range of k values to try
        max_k = min(10, len(X_dense) // 2)  # Limit upper bound
        
        # Calculate silhouette scores
        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_dense)
            score = silhouette_score(X_dense, cluster_labels)
            silhouette_scores.append(score)
        
        # Find the optimal k (highest silhouette score)
        best_k = silhouette_scores.index(max(silhouette_scores)) + 2
        return best_k

    def train(self, X):
        # Create preprocessor
        preprocessor = self._create_preprocessor(X)
        
        # If KMeans with auto k, perform silhouette analysis
        if self.cfg["alg"] == "KMeans" and self.model_kws.get('n_clusters') == -1:
            self.best_k = self.perform_silhouette_analysis(X)
            self.algorithm = KMeans(n_clusters=self.best_k)
        
        # Create pipeline with dense transformation steps for GMM
        if self.cfg["alg"] == "GMM":
            self.pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("dense_transformer", ForceDenseTransformer()),
                ("model", self.algorithm)
            ])
        else:
            self.pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", self.algorithm)
            ])
        
        # Fit the pipeline
        self.pipeline.fit(X)
        

    def process_cluster(self, X):
        # Handling different clustering algorithms
        if self.cfg["alg"] == "KMeans":
            clusters = self.pipeline.predict(X)
        elif self.cfg["alg"] == "DBSCAN":
            # DBSCAN uses fit_predict instead of predict
            clusters = self.pipeline.named_steps['model'].fit_predict(
                self.pipeline.named_steps['preprocessor'].transform(X)
            )
        elif self.cfg["alg"] == "GMM":
            clusters = self.pipeline.predict(X)
        
        # Add clusters to dataframe
        X_with_clusters = X.copy()
        X_with_clusters["cluster"] = clusters
        
        # Additional info
        additional_info = {"best_k": self.best_k} if self.best_k else None
        
        return X_with_clusters, additional_info

    def save_model(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def load_model(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


class ForceDenseTransformer:
    """Custom transformer to forcibly convert sparse matrix to dense."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, 'toarray') else X