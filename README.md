# 🐦 Flappy-Bird-Game-Trained-With-DQN

<div align="center">

![Flappy Bird DQN](https://img.shields.io/badge/Project-Flappy%20Bird%20DQN-brightgreen?style=for-the-badge&logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-0081A7?style=for-the-badge&logo=openaigym&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

<br/>

> **A Double Deep Q-Network (DDQN) reinforcement learning agent that learns to play Flappy Bird from scratch — with reward shaping, gradient clipping, and best-episode rendering.**

<br/>

<img src="https://i.pcmag.com/imagery/reviews/06fBcC3YpdFj7i0VvkWspTj-1.fit_lim.size_885x1444.v_1569469985.jpg" alt="Agent playing Flappy Bird" height="300"/>

<br/>

[![Watch Demo](https://img.shields.io/badge/▶%20Watch%20Demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](#)
[![Open Issues](https://img.shields.io/github/issues/saifullah857/Flappy-Bird-Game-Trained-With-DQN?style=for-the-badge&logo=github)](https://github.com/saifullah857/Flappy-Bird-Game-Trained-With-DQN/issues)
[![Stars](https://img.shields.io/github/stars/saifullah857/Flappy-Bird-Game-Trained-With-DQN?style=for-the-badge&logo=github&color=gold)](https://github.com/saifullah857/Flappy-Bird-Game-Trained-With-DQN/stargazers)

</div>

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [🧠 How It Works](#-how-it-works)
- [🗂️ Project Structure](#️-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [🎛️ Hyperparameters](#️-hyperparameters)
- [📊 Training Results](#-training-results)
- [🛠️ Technologies Used](#️-technologies-used)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Overview

**Flappy-Bird-Game-Trained-With-DQN** trains an autonomous AI agent to play **Flappy Bird** using **Double Deep Q-Network (DDQN)** — an improved deep reinforcement learning algorithm over standard DQN. The agent observes the game state, learns through thousands of trials, and masters pipe-dodging without ever being explicitly programmed how.

<div align="center">

| Feature | Detail |
|---|---|
| 🧠 Algorithm | Double Deep Q-Network (DDQN) |
| 🎮 Environment | `FlappyBird-v0` via `flappy-bird-gymnasium` |
| 🔁 Experience Replay | ✅ Circular buffer (200,000 capacity) |
| 🎯 Target Network | ✅ Periodic sync every 2000 steps |
| 📉 Epsilon Decay | ✅ Exponential decay (0.9998) |
| 🏆 Reward Shaping | ✅ Survival bonus + centering bonus + velocity penalty |
| ✂️ Gradient Clipping | ✅ max_norm=10 for stable training |
| 💾 Auto-Save | ✅ Best model checkpoint + all-time best |
| 🎬 Best Renderer | ✅ Silently finds best episode then renders it |
| ⚡ Device Support | CPU / CUDA / Apple MPS |

</div>

---

## 🧠 How It Works

```
Agent observes 180-dimensional state
        ↓
ε-greedy policy → Flap or No Flap
        ↓
Reward Shaping (survival + centering + velocity penalty)
        ↓
Store (state, action, reward, next_state) in Replay Buffer
        ↓
Sample mini-batch → Compute Double DQN Bellman target
        ↓
Backpropagate loss + Gradient Clipping → Update Policy Network
        ↓
Periodically sync Target Network (every 2000 steps)
        ↓
Auto-save when new best reward is achieved
```

### 🏗️ Network Architecture

```
Input Layer   (state_dim neurons)  ← Game state
      ↓
Hidden Layer  (256 neurons) + ReLU
      ↓
Hidden Layer  (256 neurons) + ReLU
      ↓
Hidden Layer  (128 neurons) + ReLU
      ↓
Output Layer  (2 neurons)          ← Q-values for [No Flap, Flap]
```

### 🔑 Key Components

| Component | Description |
|---|---|
| **Policy Network** | Makes real-time action decisions |
| **Target Network** | Provides stable Q-value targets, synced every 2000 steps |
| **Replay Memory** | Stores 200,000 past transitions; breaks temporal correlation |
| **ε-Greedy Exploration** | Starts at 1.0, decays to 0.01 over ~15,000 episodes |
| **Double DQN** | Policy net picks action, target net evaluates it — prevents Q-value overestimation |
| **Reward Shaping** | +0.1 survival/step, +0.2 centering bonus, −5.0 death penalty |
| **Gradient Clipping** | Prevents catastrophic forgetting from bad mini-batches |

---

## 🗂️ Project Structure

```
Flappy-Bird-Game-Trained-With-DQN/
│
├── 📄 agent.py               # Core DDQN agent — training & inference loop
├── 📄 dqn.py                 # Neural network definition (3 hidden layers)
├── 📄 experiance_replay.py   # Replay memory buffer
├── 📄 game_flappy_bird.py    # Play manually with keyboard input
├── 📄 render_best.py         # Find & render the best episode silently
├── 📄 parameters.yaml        # Hyperparameter configuration sets
│
└── 📁 runs/
    ├── flappybirdv0.pt        # Latest best model weights
    ├── flappybirdv0_best.pt   # All-time best model weights
    └── flappybirdv0.log       # Training reward log
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/saifullah857/Flappy-Bird-Game-Trained-With-DQN.git
cd Flappy-Bird-Game-Trained-With-DQN
```

### 2. Create a Virtual Environment *(recommended)*

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install torch gymnasium flappy-bird-gymnasium pygame pyyaml numpy
```

> 💡 For GPU support, visit [pytorch.org](https://pytorch.org/get-started/locally/) and install the CUDA-compatible build.

### 4. Fix OpenMP conflict on Windows *(if needed)*

If you get `OMP: Error #15` on Windows PowerShell:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

Run this once before any python command in the same terminal session.

---

## 🚀 Usage

### 🎮 Play Manually

Control the bird yourself using the **Spacebar**:

```bash
python game_flappy_bird.py
```

---

### 🏋️ Train the Agent

```bash
python agent.py flappybirdv0 --train
```

- Trains indefinitely until you stop with **Ctrl+C**
- Best model auto-saved to `runs/flappybirdv0_best.pt`
- Training progress logged to `runs/flappybirdv0.log`

---

### 🤖 Watch the Trained Agent Play

```bash
# Watch 10 episodes (default)
python agent.py flappybirdv0

# Watch a specific number of episodes
python agent.py flappybirdv0 --test-episodes 20
```

---

### 🏆 Find & Render the Best Episode

Silently tests N episodes in the background, finds the highest scoring one, then renders only that:

```bash
python render_best.py
```

Change `SEARCH_COUNT` inside `render_best.py` to search more episodes:

```python
SEARCH_COUNT = 500  # search 500 silent runs before rendering the best
```

---

### ➕ Add a New Hyperparameter Set

Edit `parameters.yaml` and add a new block:

```yaml
my_experiment:
  epsilon_init: 1.0
  epsilon_min: 0.01
  epsilon_decay: 0.9998
  replay_memory_size: 200000
  mini_batch_size: 64
  network_sync_rate: 2000
  alpha: 0.0005
  gamma: 0.99
  reward_threshold: 5000
```

Then train with:

```bash
python agent.py my_experiment --train
```

---

## 🎛️ Hyperparameters

Configured via `parameters.yaml`:

| Parameter | Value | Description |
|---|---|---|
| `alpha` | `0.0005` | Learning rate for Adam optimizer |
| `gamma` | `0.99` | Discount factor for future rewards |
| `epsilon_init` | `1.0` | Starting exploration rate |
| `epsilon_min` | `0.01` | Minimum exploration rate |
| `epsilon_decay` | `0.9998` | Multiplicative decay per episode |
| `replay_memory_size` | `200,000` | Max transitions stored in buffer |
| `mini_batch_size` | `64` | Sample size per training step |
| `network_sync_rate` | `2000` | Steps between target network updates |
| `reward_threshold` | `5000` | Max reward per episode before reset |

### 🎯 How Many Episodes to Train?

| Episodes | Expected Performance |
|---|---|
| 0 – 2,000 | Random / very early learning |
| 2,000 – 5,000 | Occasionally passes 1–2 pipes |
| 5,000 – 15,000 | Agent learns gap alignment |
| 15,000 – 30,000 | Consistent performance, best zone |
| 30,000+ | Diminishing returns |

> With `epsilon_decay: 0.9998`, the agent reaches near-greedy behavior at ~15,000 episodes. **Sweet spot: 20,000–30,000 episodes.**

---

## 📊 Training Results

Actual training log from this model:

```
New best reward = -7.00   at episode 1
New best reward = 2.00    at episode 2926
New best reward = 12.00   at episode 9014
New best reward = 45.00   at episode 12910
New best reward = 54.00   at episode 16417
New best reward = 77.49   at episode 18666
New best reward = 94.75   at episode 23088
New best reward = 107.26  at episode 24059
New best reward = 124.28  at episode 25301
New best reward = 143.87  at episode 26169   ← current best
```

> 📁 Full logs saved to `runs/flappybirdv0.log`

---

## 🔧 Improvements Over Basic DQN

| Improvement | Impact |
|---|---|
| **Double DQN** | Stops Q-value overestimation → better decisions near pipes |
| **Death penalty (−5.0)** | Makes survival the top priority |
| **Survival bonus (+0.1/step)** | Solves sparse reward problem early in training |
| **Centering bonus (+0.2)** | Teaches bird *where* to fly, not just whether to flap |
| **Velocity penalty (−0.05)** | Prevents overshooting the gap center |
| **Gradient clipping** | Prevents catastrophic forgetting after bad batches |
| **Deeper network (3 layers)** | Learns complex pipe-gap timing relationships |
| **Larger replay buffer (200k)** | More diverse experience, more stable learning |
| **Larger batch size (64)** | Less noisy gradient updates |
| **Lower LR (0.0005)** | Smoother, more stable convergence |

---

## 🛠️ Technologies Used

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A7?style=flat-square&logo=openaigym&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![PyGame](https://img.shields.io/badge/PyGame-3F3F3F?style=flat-square&logo=python&logoColor=white)
![YAML](https://img.shields.io/badge/YAML-CB171E?style=flat-square&logo=yaml&logoColor=white)

</div>

| Library | Purpose |
|---|---|
| `PyTorch` | Neural network & training |
| `Gymnasium` | RL environment interface |
| `flappy-bird-gymnasium` | Flappy Bird environment |
| `PyGame` | Rendering & manual play |
| `PyYAML` | Hyperparameter configuration |
| `NumPy` | Reward history & statistics |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. 🍴 Fork the repo
2. 🌿 Create a new branch: `git checkout -b feature/your-feature`
3. 💾 Commit your changes: `git commit -m 'Add some feature'`
4. 📤 Push to the branch: `git push origin feature/your-feature`
5. 📬 Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ and a lot of failed flaps.

⭐ **Star this repo if the bird finally made it through!** ⭐

</div>