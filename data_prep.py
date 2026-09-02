"""
Data preparation for the Sequential Movie Recommendation environment.
======================================================================
Loads MovieLens (ml-latest-small: ratings.csv, movies.csv), builds:
  - a reduced candidate action set (top-N most-rated movies, so the
    discrete action space stays a manageable size for DQN),
  - per-user rating histories,
  - a chronological (temporal) train / held-out test split per user, so
    evaluation happens on interactions the agent never trained on.
"""

import numpy as np
import pandas as pd

GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
GENRE_INDEX = {g: i for i, g in enumerate(GENRE_LIST)}
N_GENRES = len(GENRE_LIST)


def load_movielens(ratings_path, movies_path, n_candidate_movies=200,
                    min_user_ratings=20, test_fraction=0.2):
    """
    Returns a dict with everything the environment needs:
      movies_df           -- filtered movie metadata (candidate set only)
      genre_matrix         -- (n_candidates, N_GENRES) binary genre matrix
      movie_id_to_action   -- dict: movieId -> action index (0..n_candidates-1)
      action_to_movie_id    -- inverse mapping
      user_train            -- dict: userId -> list of (movieId, rating) in time order (train portion)
      user_test              -- dict: userId -> list of (movieId, rating) in time order (held-out portion)
      user_ids              -- list of user ids with enough history to use
    """
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # keep only users with enough ratings to have a meaningful train/test split
    counts = ratings.groupby("userId").size()
    keep_users = counts[counts >= min_user_ratings].index
    ratings = ratings[ratings["userId"].isin(keep_users)]

    # candidate action set = the N most-rated movies overall (keeps the
    # discrete action space tractable, and ensures the RL agent is choosing
    # among movies with enough rating signal to learn from)
    movie_counts = ratings.groupby("movieId").size().sort_values(ascending=False)
    candidate_ids = movie_counts.head(n_candidate_movies).index.tolist()
    candidate_ids_set = set(candidate_ids)

    movies_df = movies[movies["movieId"].isin(candidate_ids_set)].copy()
    movies_df = movies_df.set_index("movieId").loc[candidate_ids].reset_index()

    movie_id_to_action = {mid: i for i, mid in enumerate(candidate_ids)}
    action_to_movie_id = {i: mid for mid, i in movie_id_to_action.items()}

    # build binary genre matrix aligned to action index
    genre_matrix = np.zeros((len(candidate_ids), N_GENRES), dtype=np.float32)
    for i, row in movies_df.iterrows():
        action_idx = movie_id_to_action[row["movieId"]]
        for g in str(row["genres"]).split("|"):
            if g in GENRE_INDEX:
                genre_matrix[action_idx, GENRE_INDEX[g]] = 1.0

    # only keep each user's ratings that fall within the candidate set --
    # the agent can only ever recommend from this set, so only these
    # interactions are meaningful for reward / evaluation
    ratings = ratings[ratings["movieId"].isin(candidate_ids_set)]

    user_train, user_test, user_ids = {}, {}, []
    for uid, group in ratings.groupby("userId"):
        group = group.sort_values("timestamp")
        interactions = list(zip(group["movieId"].tolist(), group["rating"].tolist()))
        if len(interactions) < 10:
            continue  # not enough signal within the candidate set for this user
        split_point = int(len(interactions) * (1 - test_fraction))
        split_point = max(split_point, 5)  # ensure some train history
        train_part = interactions[:split_point]
        test_part = interactions[split_point:]
        if len(test_part) == 0:
            continue
        user_train[uid] = train_part
        user_test[uid] = test_part
        user_ids.append(uid)

    return {
        "movies_df": movies_df,
        "genre_matrix": genre_matrix,
        "movie_id_to_action": movie_id_to_action,
        "action_to_movie_id": action_to_movie_id,
        "user_train": user_train,
        "user_test": user_test,
        "user_ids": user_ids,
        "n_actions": len(candidate_ids),
    }


if __name__ == "__main__":
    data = load_movielens("data/ratings.csv", "data/movies.csv")
    print(f"Candidate movies (actions): {data['n_actions']}")
    print(f"Usable users: {len(data['user_ids'])}")
    example_uid = data["user_ids"][0]
    print(f"Example user {example_uid}: "
          f"{len(data['user_train'][example_uid])} train interactions, "
          f"{len(data['user_test'][example_uid])} test interactions")
