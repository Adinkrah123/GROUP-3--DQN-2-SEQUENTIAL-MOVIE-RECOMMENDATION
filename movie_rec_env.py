"""
Sequential Movie Recommendation Environment (Gymnasium API)
=============================================================
Project DQN-2: at each step, the agent recommends one movie (from a
reduced candidate set of the N most-rated movies) to a simulated user.
The "user" is simulated using that user's real, held-out interaction
history from MovieLens: if the recommended movie is one the user
actually rated, the reward reflects how much they liked it; if the
movie was never rated by this user, a small exploration penalty is
given. Already-recommended movies in the same episode are masked out
(cannot be repeated).

State (observation) -- all continuous / normalised, fixed-size vector:
  - user genre-preference profile (N_GENRES,): mean normalised rating
    the user has given to movies in each genre, from their training history
  - recent recommendation history (HISTORY_LEN,): the last few
    recommended movie's genre vectors, flattened (so the agent can avoid
    recommending too repetitively within a genre)
  - normalised step count within the episode

Action: Discrete(n_candidate_movies) -- pick a movie to recommend
        (invalid / already-recommended actions are masked)

Reward:
  - if the user actually rated this movie in their available history:
        reward = (rating - 3) / 2       (maps 0.5..5.0 stars to roughly -1.25..1.0,
                                          positive above the "3-star" satisfaction threshold)
  - if the user never rated this movie (unknown preference):
        reward = -0.1                    (small penalty; not enough signal to reward exploration)
  - repeating an already-recommended movie in the same episode: reward = -1.0 (heavily discouraged)

Episode ends after EPISODE_LENGTH recommendations (or when the candidate
list is exhausted, whichever comes first).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from data_prep import N_GENRES

EPISODE_LENGTH = 10
HISTORY_LEN = 3  # number of recent recommendations remembered in the state
REPEAT_PENALTY = -1.0
UNKNOWN_PENALTY = -0.1


class MovieRecEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data, user_split="train", seed=None):
        """
        data       -- dict returned by data_prep.load_movielens
        user_split -- "train" or "test": which interaction split to simulate
                      the user's preferences from (use "test" for held-out evaluation)
        """
        super().__init__()
        self.data = data
        self.user_split = user_split
        self.genre_matrix = data["genre_matrix"]  # (n_actions, N_GENRES)
        self.n_actions = data["n_actions"]
        self.user_ids = data["user_ids"]

        self.action_space = spaces.Discrete(self.n_actions)

        obs_dim = N_GENRES + HISTORY_LEN * N_GENRES + 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._np_random = np.random.default_rng(seed)
        self.current_user = None
        self.user_ratings_lookup = None
        self.genre_profile = None
        self.recommended_this_episode = None
        self.history_buffer = None
        self.t = 0

    def _build_genre_profile(self, interactions):
        """Mean normalised rating (rating-3)/2 per genre, from a user's interaction list."""
        profile = np.zeros(N_GENRES, dtype=np.float32)
        counts = np.zeros(N_GENRES, dtype=np.float32)
        action_map = self.data["movie_id_to_action"]
        for movie_id, rating in interactions:
            a = action_map.get(movie_id)
            if a is None:
                continue
            norm_rating = (rating - 3.0) / 2.0
            genres = self.genre_matrix[a]
            profile += genres * norm_rating
            counts += genres
        counts = np.maximum(counts, 1e-6)
        return profile / counts

    def _get_obs(self):
        history_flat = self.history_buffer.flatten()
        step_norm = np.array([self.t / EPISODE_LENGTH], dtype=np.float32)
        obs = np.concatenate([self.genre_profile, history_flat, step_norm]).astype(np.float32)
        return obs

    def _action_mask(self):
        mask = np.ones(self.n_actions, dtype=bool)
        mask[list(self.recommended_this_episode)] = False
        return mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # pick a user for this episode
        user_idx = self._np_random.integers(0, len(self.user_ids))
        self.current_user = self.user_ids[user_idx]

        train_interactions = self.data["user_train"][self.current_user]
        eval_interactions = self.data[f"user_{self.user_split}"][self.current_user]

        # genre profile is always built from TRAIN history (what the agent
        # is "told" about the user); the reward signal (ground truth) comes
        # from the requested split (train during training, test during eval)
        self.genre_profile = self._build_genre_profile(train_interactions)
        self.user_ratings_lookup = {mid: r for mid, r in eval_interactions}

        self.recommended_this_episode = set()
        self.history_buffer = np.zeros((HISTORY_LEN, N_GENRES), dtype=np.float32)
        self.t = 0

        obs = self._get_obs()
        info = {"action_mask": self._action_mask(), "user_id": self.current_user}
        return obs, info

    def step(self, action):
        action = int(action)
        repeated = action in self.recommended_this_episode

        if repeated:
            reward = REPEAT_PENALTY
            hit = False
            rating_used = None
        else:
            movie_id = self.data["action_to_movie_id"][action]
            if movie_id in self.user_ratings_lookup:
                rating = self.user_ratings_lookup[movie_id]
                reward = (rating - 3.0) / 2.0
                hit = rating >= 4.0
                rating_used = rating
            else:
                reward = UNKNOWN_PENALTY
                hit = False
                rating_used = None

            self.recommended_this_episode.add(action)
            # update recent-history buffer (shift, insert newest)
            self.history_buffer = np.roll(self.history_buffer, shift=-1, axis=0)
            self.history_buffer[-1] = self.genre_matrix[action]

        self.t += 1
        terminated = False
        truncated = self.t >= EPISODE_LENGTH or len(self.recommended_this_episode) >= self.n_actions

        obs = self._get_obs()
        info = {
            "action_mask": self._action_mask(),
            "hit": hit,
            "repeated": repeated,
            "rating": rating_used,
            "user_id": self.current_user,
        }
        return obs, float(reward), terminated, truncated, info
