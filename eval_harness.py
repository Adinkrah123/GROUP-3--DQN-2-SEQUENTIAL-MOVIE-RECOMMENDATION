"""
Evaluation harness -- shared across the DQN agent and both baselines.
========================================================================
Runs a policy (any callable that maps an observation/info to an action)
through held-out ("test" split) episodes for a fixed set of users and
random seeds, and computes:
    - cumulative reward
    - hit_rate@K       : fraction of episodes with at least one "hit"
                          (a recommended movie the user rated >= 4 stars)
    - precision@K       : fraction of the K recommended movies that were hits
    - NDCG@K              : rewards hits that occur earlier in the list more
    - intra-list diversity (ILD): average pairwise genre dissimilarity
                          among the K recommended movies (higher = more diverse)

Every policy is evaluated on IDENTICAL episodes (same users, same seeds,
same environment reset behaviour), per the exam's baseline-comparison
requirement.
"""

import numpy as np

from movie_rec_env import MovieRecEnv, EPISODE_LENGTH


def _ndcg_at_k(hits):
    """hits: list of 0/1 in recommendation order."""
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal_hits = sorted(hits, reverse=True)
    idcg = sum(h / np.log2(i + 2) for i, h in enumerate(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _intra_list_diversity(genre_vectors):
    """Average pairwise cosine DIStance between recommended movies' genre vectors."""
    n = len(genre_vectors)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = genre_vectors[i], genre_vectors[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-8 or nb < 1e-8:
                sim = 0.0
            else:
                sim = float(np.dot(a, b) / (na * nb))
            total += (1 - sim)
            count += 1
    return total / count


def evaluate_policy(policy_fn, data, seeds, n_episodes_per_seed=30, needs_user_id=False):
    """
    policy_fn(obs, action_mask, user_id) -> action   (all policies share this signature;
                                                         DQN policies can ignore user_id)
    Returns a dict of per-seed metric lists, ready for mean/std aggregation.
    """
    env = MovieRecEnv(data, user_split="test")

    per_seed_metrics = {
        "cumulative_reward": [],
        "hit_rate": [],
        "precision_at_k": [],
        "ndcg_at_k": [],
        "diversity": [],
    }

    for seed in seeds:
        seed_rewards, seed_hit_rate, seed_precision, seed_ndcg, seed_diversity = [], [], [], [], []
        rng_seed = seed * 10_000  # spread out episode seeds deterministically per outer seed
        for ep in range(n_episodes_per_seed):
            obs, info = env.reset(seed=rng_seed + ep)
            action_mask = info["action_mask"]
            user_id = info["user_id"]

            ep_reward = 0.0
            hits = []
            genre_vecs = []
            done = False
            while not done:
                action = policy_fn(obs, action_mask, user_id)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                hits.append(1 if info["hit"] else 0)
                genre_vecs.append(env.genre_matrix[action].copy())
                action_mask = info["action_mask"]
                done = terminated or truncated

            seed_rewards.append(ep_reward)
            seed_hit_rate.append(1.0 if sum(hits) > 0 else 0.0)
            seed_precision.append(sum(hits) / len(hits))
            seed_ndcg.append(_ndcg_at_k(hits))
            seed_diversity.append(_intra_list_diversity(genre_vecs))

        per_seed_metrics["cumulative_reward"].append(np.mean(seed_rewards))
        per_seed_metrics["hit_rate"].append(np.mean(seed_hit_rate))
        per_seed_metrics["precision_at_k"].append(np.mean(seed_precision))
        per_seed_metrics["ndcg_at_k"].append(np.mean(seed_ndcg))
        per_seed_metrics["diversity"].append(np.mean(seed_diversity))

    return per_seed_metrics


def summarize(per_seed_metrics):
    """Returns {metric: (mean_across_seeds, std_across_seeds)}."""
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in per_seed_metrics.items()}


def print_summary(name, summary):
    print(f"\n=== {name} ===")
    for metric, (mean, std) in summary.items():
        print(f"  {metric:<20s}: {mean:8.4f} +/- {std:.4f}")


if __name__ == "__main__":
    from data_prep import load_movielens
    from baseline import PopularityRecommender, ItemCFRecommender

    data = load_movielens("data/ratings.csv", "data/movies.csv")
    SEEDS = [0, 1, 2]

    # --- Popularity baseline ---
    pop = PopularityRecommender(data)
    pop_metrics = evaluate_policy(
        lambda obs, mask, uid: pop.recommend(mask), data, SEEDS
    )
    print_summary("Popularity Baseline", summarize(pop_metrics))

    # --- Item-based CF baseline ---
    cf = ItemCFRecommender(data)
    cf_metrics = evaluate_policy(
        lambda obs, mask, uid: cf.recommend(mask, uid), data, SEEDS
    )
    print_summary("Item-based Collaborative Filtering Baseline", summarize(cf_metrics))

    # --- Random policy (sanity lower-bound) ---
    rng = np.random.default_rng(42)
    random_metrics = evaluate_policy(
        lambda obs, mask, uid: rng.choice(np.where(mask)[0]), data, SEEDS
    )
    print_summary("Random Policy (sanity check)", summarize(random_metrics))
