import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
import torch
from experiance_replay import ReplayMemory
import yaml
import torch.nn as nn
import itertools
import random
import torch.optim as optim
import argparse
import os
import numpy as np

# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_sets = yaml.safe_load(f)

        if param_set not in all_param_sets:
            raise ValueError(f"Parameter set '{param_set}' not found in parameters.yaml")

        params = all_param_sets[param_set]

        self.alpha              = params["alpha"]
        self.gamma              = params["gamma"]
        self.epsilon_init       = params["epsilon_init"]
        self.epsilon_min        = params["epsilon_min"]
        self.epsilon_decay      = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size    = params["mini_batch_size"]
        self.reward_threshold   = params["reward_threshold"]
        self.network_sync_rate  = params["network_sync_rate"]

        self.loss_fn   = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE        = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE      = os.path.join(RUNS_DIR, f"{self.param_set}.pt")
        self.BEST_MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}_best.pt")

    # ------------------------------------------------------------------
    # Reward Shaping
    # Observation indices (flappy_bird_gymnasium v0):
    #   [0]  last pipe horizontal pos
    #   [1]  last top pipe vertical pos
    #   [2]  last bottom pipe vertical pos
    #   [3]  next pipe horizontal pos
    #   [4]  next top pipe vertical pos
    #   [5]  next bottom pipe vertical pos
    #   [6]  next-next pipe horizontal pos
    #   [7]  next-next top pipe vertical pos
    #   [8]  next-next bottom pipe vertical pos
    #   [9]  player vertical pos
    #   [10] player vertical velocity
    #   [11] player rotation
    # ------------------------------------------------------------------
    def shape_reward(self, reward, state, terminated):
        # Hard death penalty — survival is the top priority
        if terminated:
            reward -= 5.0
            return reward

        # 1. Survival bonus — reward every step alive
        reward += 0.1

        # 2. Centering bonus — quadratic, smooth falloff toward gap edges
        player_y      = state[9].item()
        next_top_pipe = state[4].item()
        next_bot_pipe = state[5].item()
        gap_center    = (next_top_pipe + next_bot_pipe) / 2.0
        gap_half      = abs(next_top_pipe - next_bot_pipe) / 2.0

        if gap_half > 0:
            norm_dist       = abs(player_y - gap_center) / gap_half
            centering_bonus = max(0.0, 0.2 * (1.0 - norm_dist ** 2))
            reward         += centering_bonus

        # 3. Velocity penalty — punish moving away from gap center
        velocity = state[10].item()
        if (player_y < gap_center and velocity < -2.0) or \
           (player_y > gap_center and velocity > 2.0):
            reward -= 0.05

        return reward

    def run(self, is_training=True, render=False, num_test_episodes=10):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states  = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory         = ReplayMemory(self.replay_memory_size)
            epsilon        = self.epsilon_init
            target_dqn     = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps          = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward    = float("-inf")
            reward_history = []

        else:
            # Always load best saved model for testing
            model_to_load = self.BEST_MODEL_FILE if os.path.exists(self.BEST_MODEL_FILE) \
                            else self.MODEL_FILE

            if not os.path.exists(model_to_load):
                raise FileNotFoundError(
                    f"No saved model found at {model_to_load}. Train first with --train."
                )

            print(f"Loading model from: {model_to_load}")
            policy_dqn.load_state_dict(
                torch.load(model_to_load, map_location=device, weights_only=True)
            )
            policy_dqn.eval()
            test_rewards = []

        for episode in itertools.count():

            # Stop test mode after N episodes and show summary
            if not is_training and episode >= num_test_episodes:
                print("\n========== Test Summary ==========")
                print(f"Episodes  : {num_test_episodes}")
                print(f"Avg Reward: {np.mean(test_rewards):.2f}")
                print(f"Max Reward: {np.max(test_rewards):.2f}")
                print(f"Min Reward: {np.min(test_rewards):.2f}")
                print(f"Std Dev   : {np.std(test_rewards):.2f}")
                print("==================================")
                break

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            episode_reward = 0.0
            terminated     = False
            truncated      = False

            while not terminated and not truncated and episode_reward < self.reward_threshold:
                # Epsilon-greedy action selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze(0).argmax()

                next_state, reward, terminated, truncated, _ = env.step(action.item())

                # Shape reward during training only
                if is_training:
                    reward = self.shape_reward(
                        reward,
                        torch.tensor(next_state, dtype=torch.float32, device=device),
                        terminated
                    )

                reward     = torch.tensor(reward,     dtype=torch.float32, device=device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                state          = next_state
                episode_reward += reward.item()

            # ---- Logging ----
            if is_training:
                reward_history.append(episode_reward)
                avg_reward = np.mean(reward_history[-100:])
                print(
                    f"Episode {episode + 1:>6} | "
                    f"Reward: {episode_reward:>10.2f} | "
                    f"Avg(100): {avg_reward:>10.2f} | "
                    f"Epsilon: {epsilon:.5f}"
                )
            else:
                test_rewards.append(episode_reward)
                print(f"Test Episode {episode + 1}/{num_test_episodes} | Reward: {episode_reward:.2f}")

            # ---- Training updates ----
            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # Save best model
                if episode_reward > best_reward:
                    log_msg = f"New best reward = {episode_reward:.2f} at episode {episode + 1}"
                    print(f"  *** {log_msg} ***")
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    torch.save(policy_dqn.state_dict(), self.BEST_MODEL_FILE)
                    best_reward = episode_reward

                # Optimize
                if len(memory) >= self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                # Sync target network
                if steps >= self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states       = torch.stack(states)
        actions      = torch.stack(actions).long()
        next_states  = torch.stack(next_states)
        rewards      = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float32, device=device)

        # Double DQN: policy network picks best action, target network evaluates it
        with torch.no_grad():
            best_actions = policy_dqn(next_states).argmax(dim=1, keepdim=True)
            target_q     = rewards + (1 - terminations) * self.gamma * \
                           target_dqn(next_states).gather(1, best_actions).squeeze(1)

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents catastrophic forgetting
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=10)
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test FlappyBird RL agent")
    parser.add_argument("hyperparameters", help="Name of parameter set in parameters.yaml")
    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument(
        "--test-episodes", type=int, default=10,
        help="Number of episodes to run in test mode (default: 10)"
    )
    args = parser.parse_args()

    agent = Agent(param_set=args.hyperparameters)

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True, num_test_episodes=args.test_episodes)