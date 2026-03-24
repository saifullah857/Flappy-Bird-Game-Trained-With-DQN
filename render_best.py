import torch
import gymnasium as gym
import flappy_bird_gymnasium
from dqn import DQN
import random

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

MODEL_FILE   = "runs/flappybirdv0_best.pt"
SEARCH_COUNT = 100  # check 100 episodes silently then render the best one

silent_env = gym.make("FlappyBird-v0", render_mode=None)
render_env = gym.make("FlappyBird-v0", render_mode="human")

num_states  = silent_env.observation_space.shape[0]
num_actions = silent_env.action_space.n

policy_dqn  = DQN(num_states, num_actions).to(device)
policy_dqn.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
policy_dqn.eval()

def run_episode(env, seed):
    state, _       = env.reset(seed=seed)
    state          = torch.tensor(state, dtype=torch.float32, device=device)
    episode_reward = 0.0
    terminated     = False
    truncated      = False

    while not terminated and not truncated:
        with torch.no_grad():
            action = policy_dqn(state.unsqueeze(0)).squeeze(0).argmax()
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        state          = torch.tensor(next_state, dtype=torch.float32, device=device)
        episode_reward += reward

    return episode_reward

print(f"Searching best episode out of {SEARCH_COUNT} silent runs...\n")

best_reward = float("-inf")
best_seed   = None

for i in range(SEARCH_COUNT):
    seed   = random.randint(0, 999999)
    reward = run_episode(silent_env, seed)
    print(f"  [{i+1}/{SEARCH_COUNT}] Reward: {reward:.2f}" + (" ← new best!" if reward > best_reward else ""))

    if reward > best_reward:
        best_reward = reward
        best_seed   = seed

print(f"\nBest reward found: {best_reward:.2f}")
print(f"Rendering best episode now...\n")

run_episode(render_env, best_seed)

print(f"Done! Best reward was: {best_reward:.2f}")