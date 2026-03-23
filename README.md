# 🐦 Flappy-Bird-Game-Trained-With-DQN

<div align="center">

![Flappy Bird DQN](https://img.shields.io/badge/Project-Flappy%20Bird%20DQN-brightgreen?style=for-the-badge&logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-0081A7?style=for-the-badge&logo=openaigym&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

<br/>

> **A Deep Q-Network (DQN) reinforcement learning agent that learns to play Flappy Bird from scratch — no human input, just raw trial and error.**

<br/>


   . <img src="https://i.pcmag.com/imagery/reviews/06fBcC3YpdFj7i0VvkWspTj-1.fit_lim.size_885x1444.v_1569469985.jpg" alt="Agent playing Flappy Bird" height="300"/>
    


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

**Flappy-Bird-Game-Trained-With-DQN** trains an autonomous AI agent to play **Flappy Bird** using **Deep Q-Network (DQN)** — a foundational deep reinforcement learning algorithm. The agent observes the game state, learns through millions of trials, and eventually masters pipe-dodging without ever being explicitly programmed how.

<div align="center">

| Feature | Detail |
|---|---|
| 🧠 Algorithm | Deep Q-Network (DQN) |
| 🎮 Environment | `FlappyBird-v0` via `flappy-bird-gymnasium` |
| 🔁 Experience Replay | ✅ Circular buffer |
| 🎯 Target Network | ✅ Periodic sync |
| 📉 Epsilon Decay | ✅ Exponential decay |
| 💾 Auto-Save | ✅ Best model checkpoint |
| ⚡ Device Support | CPU / CUDA / Apple MPS |

</div>

---

## 🧠 How It Works

```
Agent observes 12-dimensional state
        ↓
ε-greedy policy → Flap or No Flap
        ↓
Store (state, action, reward, next_state) in Replay Buffer
        ↓
Sample mini-batch → Compute Bellman target
        ↓
Backpropagate loss → Update Policy Network
        ↓
Periodically sync Target Network
```

### 🏗️ Network Architecture

```
Input Layer  (12 neurons)  ← Game state
      ↓
Hidden Layer (250 neurons) + ReLU
      ↓
Output Layer (2 neurons)   ← Q-values for [No Flap, Flap]
```

### 🔑 Key DQN Components

| Component | Description |
|---|---|
| **Policy Network** | Makes real-time action decisions |
| **Target Network** | Provides stable Q-value targets, synced every N steps |
| **Replay Memory** | Stores past transitions; breaks temporal correlation |
| **ε-Greedy Exploration** | Starts random, gradually shifts to exploitation |

---

## 🗂️ Project Structure

```
Flappy-Bird-Game-Trained-With-DQN/
│
├── 📄 agent.py               # Core DQN agent — training & inference loop
├── 📄 dqn.py                 # Neural network definition (PyTorch)
├── 📄 experiance_replay.py   # Replay memory buffer
├── 📄 game_flappy_bird.py    # Play manually with keyboard input
├── 📄 parameters.yaml        # Hyperparameter configuration sets
│
└── 📁 runs/
    ├── flappybirdv0.pt       # Saved model weights (best reward)
    └── flappybirdv0.log      # Training reward log
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
pip install torch gymnasium flappy-bird-gymnasium pygame pyyaml
```

> 💡 For GPU support, visit [pytorch.org](https://pytorch.org/get-started/locally/) and install the CUDA-compatible build.

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

- The agent will train indefinitely until you stop it.
- The **best model** is auto-saved to `runs/flappybirdv0.pt`.
- Training progress is logged to `runs/flappybirdv0.log`.

---

### 🤖 Watch the Trained Agent Play

```bash
python agent.py flappybirdv0
```

> Make sure `runs/flappybirdv0.pt` exists (i.e., you've trained at least once).

---

### ➕ Add a New Hyperparameter Set

Edit `parameters.yaml` and add a new block:

```yaml
my_experiment:
  epsilon_init: 1
  epsilon_min: 0.01
  epsilon_decay: 0.999
  replay_memory_size: 50000
  mini_batch_size: 64
  network_sync_rate: 20
  alpha: 0.0005
  gamma: 0.95
  reward_threshold: 500
```

Then train with:

```bash
python agent.py my_experiment --train
```

---

## 🎛️ Hyperparameters

Configured via `parameters.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `alpha` | `0.001` | Learning rate for Adam optimizer |
| `gamma` | `0.99` | Discount factor for future rewards |
| `epsilon_init` | `1.0` | Starting exploration rate |
| `epsilon_min` | `0.05` | Minimum exploration rate |
| `epsilon_decay` | `0.9995` | Multiplicative decay per episode |
| `replay_memory_size` | `100,000` | Max transitions stored in buffer |
| `mini_batch_size` | `32` | Sample size per training step |
| `network_sync_rate` | `10` | Steps between target network updates |
| `reward_threshold` | `1000` | Max reward per episode before reset |

---

## 📊 Training Results

The agent's reward improves significantly as epsilon decays and the replay buffer fills up. Typical training curve:

```
Episode 1      Reward = -2.10   Epsilon = 1.0000
Episode 50     Reward = -1.40   Epsilon = 0.9753
Episode 200    Reward =  0.30   Epsilon = 0.9050
Episode 500    Reward =  5.20   Epsilon = 0.7790
Episode 1000   Reward = 18.50   Epsilon = 0.6068
Episode 3000   Reward = 84.70   Epsilon = 0.2231
...
```

> 📁 Full logs saved to `runs/flappybirdv0.log`

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