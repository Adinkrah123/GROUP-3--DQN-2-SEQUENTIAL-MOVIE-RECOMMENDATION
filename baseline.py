"""
Baseline recommenders for comparison against the trained DQN agent.
=====================================================================
1. PopularityRecommender -- always recommends the most-rated movies
   (by number of ratings in the training data), skipping repeats.
2. ItemCFRecommender -- item-based collaborative filtering: builds a
   user-item rating matrix from training data, computes cosine
   similarity between candidate movies, and for each recommendation
   step scores every not-yet-recommended candidate by its similarity
   to the movies the current user rated highly in their training
   history, recommending the highest-scoring one each step.

Both expose a common interface: `recommend(state_info) -> action`, so
they can be plugged into the same evaluation harness used for the DQN
agent, per the "same code path" requirement in the exam brief.
"""

import numpy as np


class PopularityRecommender:
    """Recommends movies in order of overall popularity (rating count) in the training data."""

    def __init__(self, data):
        movie_id_to_action = data["movie_id_to_action"]
        counts = np.zeros(data["n_actions"], dtype=np.int64)
        for interactions in data["user_train"].values():
            for movie_id, _ in interactions:
                a = movie_id_to_action.get(movie_id)
                if a is not None:
                    counts[a] += 1
        # actions ranked from most to least popular
        self.ranked_actions = np.argsort(-counts)

    def recommend(self, action_mask):
        for a in self.ranked_actions:
            if action_mask[a]:
                return int(a)
        # fallback (shouldn't happen: mask should always have a free action)
        return int(np.where(action_mask)[0][0])


class ItemCFRecommender:
    """Item-based collaborative filtering using cosine similarity on the training rating matrix."""

    def __init__(self, data):
        self.data = data
        n_actions = data["n_actions"]
        movie_id_to_action = data["movie_id_to_action"]
        user_ids = data["user_ids"]
        user_index = {uid: i for i, uid in enumerate(user_ids)}

        # build a (n_users, n_actions) training rating matrix
        rating_matrix = np.zeros((len(user_ids), n_actions), dtype=np.float32)
        for uid, interactions in data["user_train"].items():
            ui = user_index[uid]
            for movie_id, rating in interactions:
                a = movie_id_to_action.get(movie_id)
                if a is not None:
                    rating_matrix[ui, a] = rating

        # item-item cosine similarity matrix (n_actions, n_actions)
        norms = np.linalg.norm(rating_matrix, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = rating_matrix / norms
        self.similarity = normalized.T @ normalized  # (n_actions, n_actions)

        self.user_index = user_index
        self.rating_matrix = rating_matrix

    def recommend(self, action_mask, user_id):
        ui = self.user_index[user_id]
        user_ratings = self.rating_matrix[ui]  # (n_actions,)
        liked_mask = user_ratings >= 4.0
        if not liked_mask.any():
            liked_mask = user_ratings > 0  # fall back to anything rated

        # score each candidate by summed similarity to the user's liked movies
        scores = self.similarity[:, liked_mask].sum(axis=1)
        scores = np.where(action_mask, scores, -np.inf)
        return int(np.argmax(scores))
