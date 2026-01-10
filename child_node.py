import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import time

class ChildNode:
    def __init__(self, rank):
        self.rank = rank
        # Assign different models based on rank
        if self.rank == 1:
            self.model_name = "Naive Bayes"
            self.model = make_pipeline(CountVectorizer(), MultinomialNB())
        elif self.rank == 2:
            self.model_name = "Logistic Regression"
            self.model = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000, n_jobs=-1))
        else:
            self.model_name = "Unknown"
            self.model = None

    def train(self, data):
        """
        Receives data (texts, labels) and performs training/multiprocessing
        """
        if not self.model:
            return {"error": "No model assigned"}

        texts, labels = data
        
        print(f"[Rank {self.rank}] Starting training for {self.model_name}...")
        start_time = time.time()

        # Simulate internal parallelism using cross_val_score with n_jobs=-1
        # This satisfies the requirement for "internal parallelism"
        scores = cross_val_score(self.model, texts, labels, cv=5, n_jobs=-1)
        accuracy = np.mean(scores)
        
        # Train on full set for final model
        self.model.fit(texts, labels)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"[Rank {self.rank}] Finished. Accuracy: {accuracy:.4f}, Time: {elapsed_time:.4f}s")
        
        return {
            "rank": self.rank,
            "model": self.model_name,
            "accuracy": accuracy,
            "time": elapsed_time
        }
