"""
Train a DQN agent (Stable-Baselines3) on the Sequential Movie Recommendation
environment, across multiple seeds, and evaluate against the required
baselines using the shared evaluation harness.

Run this on Google Colab or any machine with PyTorch + Stable-Baselines3
installed (see requirements.txt) -- it needs more disk/compute than a
minimal sandbox typically provides.

Usage:
    python train_dqn.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from data_prep import load_movielens
from movie_rec_env import MovieRecEnv
from baseline import PopularityRecommender, ItemCFRecommender
from eval_harness import evaluate_policy, summarize, print_summary

# ---------------------------------------------------------------------
# Config -- report every value that differs from the SB3 default here
# ---------------------------------------------------------------------
SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 60_000          # reduce if compute-limited; state the reduction in the report
EVAL_EPISODES_PER_SEED = 30

HYPERPARAMS = dict(
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.95,                    # short (10-step) episodes -> lower discount than SB3 default 0.99
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[128, 128]),
)

OUT_DIR = "runs"
os.makedirs(OUT_DIR, exist_ok=True)


def masked_dqn_policy(model):
    """Wrap a trained SB3 DQN model into the (obs, action_mask, user_id) -> action
    signature used by eval_harness, applying HARD action masking over the
    learned Q-values (so the DQN agent never repeats a recommendation,
    matching what both baselines already guarantee by construction)."""
    import torch

    def policy_fn(obs, action_mask, user_id):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        with torch.no_grad():
            q_values = model.q_net(obs_t).cpu().numpy()[0]
        q_values = np.where(action_mask, q_values, -np.inf)
        return int(np.argmax(q_values))

    return policy_fn


def train_one_seed(data, seed):
    env = Monitor(MovieRecEnv(data, user_split="train", seed=seed))
    model = DQN("MlpPolicy", env, seed=seed, verbose=0, **HYPERPARAMS)

    log_path = os.path.join(OUT_DIR, f"seed_{seed}")
    os.makedirs(log_path, exist_ok=True)
    new_logger = configure(log_path, ["csv"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(OUT_DIR, f"dqn_movierec_seed{seed}"))

    # save the raw episode-reward log used to produce the training-curve figure
    ep_rewards = np.array(env.get_episode_rewards())
    np.save(os.path.join(log_path, "episode_rewards.npy"), ep_rewards)

    return model, ep_rewards


def main():
    print("Loading MovieLens data...")
    data = load_movielens("data/ratings.csv", "data/movies.csv")
    print(f"  {data['n_actions']} candidate movies, {len(data['user_ids'])} usable users")

    all_ep_rewards = []
    trained_models = []
    for seed in SEEDS:
        print(f"\n=== Training DQN, seed {seed} ===")
        model, ep_rewards = train_one_seed(data, seed)
        trained_models.append(model)
        all_ep_rewards.append(ep_rewards)
        print(f"  final 20-episode mean reward: {ep_rewards[-20:].mean():.3f}")

    # ---- training curve (mean +/- std across seeds) ----
    min_len = min(len(r) for r in all_ep_rewards)
    trimmed = np.stack([r[:min_len] for r in all_ep_rewards])
    window = 20
    smoothed = np.array([
        np.convolve(trimmed[i], np.ones(window) / window, mode="valid")
        for i in range(len(SEEDS))
    ])
    mean_curve = smoothed.mean(axis=0)
    std_curve = smoothed.std(axis=0)

    plt.figure(figsize=(9, 4))
    plt.plot(mean_curve, label="mean across seeds")
    plt.fill_between(range(len(mean_curve)), mean_curve - std_curve, mean_curve + std_curve,
                      alpha=0.25, label="+/- 1 std")
    plt.xlabel(f"Training episode ({window}-episode moving average)")
    plt.ylabel("Episode reward")
    plt.title(f"DQN Training Reward -- Sequential Movie Recommendation ({len(SEEDS)} seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=150)
    print(f"\nSaved training curve to {OUT_DIR}/training_curve.png")

    # ---- evaluation: DQN (all seeds) vs. both baselines, same harness ----
    print("\nEvaluating DQN across seeds on held-out test users...")
    dqn_metrics_per_seed = {k: [] for k in ["cumulative_reward", "hit_rate", "precision_at_k", "ndcg_at_k", "diversity"]}
    for seed, model in zip(SEEDS, trained_models):
        policy_fn = masked_dqn_policy(model)
        m = evaluate_policy(policy_fn, data, seeds=[seed], n_episodes_per_seed=EVAL_EPISODES_PER_SEED)
        for k in dqn_metrics_per_seed:
            dqn_metrics_per_seed[k].append(m[k][0])
    dqn_summary = summarize(dqn_metrics_per_seed)
    print_summary("DQN Agent (ours)", dqn_summary)

    print("\nEvaluating baselines (same episodes/seeds/metric code)...")
    pop = PopularityRecommender(data)
    pop_metrics = evaluate_policy(lambda obs, mask, uid: pop.recommend(mask), data, SEEDS, EVAL_EPISODES_PER_SEED)
    pop_summary = summarize(pop_metrics)
    print_summary("Popularity Baseline", pop_summary)

    cf = ItemCFRecommender(data)
    cf_metrics = evaluate_policy(lambda obs, mask, uid: cf.recommend(mask, uid), data, SEEDS, EVAL_EPISODES_PER_SEED)
    cf_summary = summarize(cf_metrics)
    print_summary("Item-based Collaborative Filtering Baseline", cf_summary)

    # ---- save everything needed to reproduce the report's figures/tables ----
    results = {
        "hyperparameters": {**HYPERPARAMS, "total_timesteps": TOTAL_TIMESTEPS, "seeds": SEEDS},
        "dqn": dqn_summary,
        "popularity_baseline": pop_summary,
        "item_cf_baseline": cf_summary,
    }
    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results summary to {OUT_DIR}/results_summary.json")


if __name__ == "__main__":
    main()
