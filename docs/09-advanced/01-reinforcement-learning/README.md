# 🎮 強化學習 | Reinforcement Learning (RL)

> 摘要：強化學習通過智能體與環境的交互學習最優策略。本章介紹 MDP 基礎、經典算法 (Q-Learning、Policy Gradient、PPO)、深度 RL (DQN、Actor-Critic) 以及大語言模型中的 RLHF。  
> Abstract: Reinforcement learning trains agents to make optimal decisions through environment interaction. This chapter covers MDP foundations, classic algorithms, deep RL, and RLHF for LLMs.

---

## 1. 強化學習基本概念 | Core Concepts

### 1.1 智能體與環境 | Agent & Environment

```
┌──────────────────────────────────────────────────────────────┐
│                     RL 框架                                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐                      ┌──────────┐            │
│   │          │  action a_t           │          │            │
│   │  Agent   │ ─────────────────→   │   Env    │            │
│   │  (智能體) │                      │ (環境)   │            │
│   │          │ ←─────────────────   │          │            │
│   └──────────┘  state s_{t+1}        └──────────┘            │
│                 reward r_{t+1}                                │
│                                                              │
│   目標：學習策略 π(a|s) 使累計獎勵最大化                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 關鍵術語 | Key Terminology

| 術語 | 英文 | 描述 |
|------|------|------|
| **狀態** | State $s$ | 環境的當前情況 |
| **動作** | Action $a$ | 智能體可以執行的操作 |
| **獎勵** | Reward $r$ | 環境對動作的即時反饋 |
| **策略** | Policy $\pi(a\|s)$ | 從狀態到動作的映射 |
| **價值** | Value $V(s)$ | 從狀態 $s$ 開始的期望累計獎勵 |
| **回報** | Return $G_t$ | $r_{t+1} + \gamma r_{t+2} + ...$ |
| **折扣因子** | Discount $\gamma$ | 對未來獎勵的重視程度，$\gamma \in [0, 1]$ |

---

## 2. 馬可夫決策過程 | Markov Decision Process (MDP)

### 2.1 定義

MDP 用元組 $(S, A, P, R, \gamma)$ 描述：

- $S$：狀態空間
- $A$：動作空間
- $P(s'|s,a)$：轉移機率
- $R(s,a,s')$：獎勵函數
- $\gamma$：折扣因子

### 2.2 馬可夫性質

未來只取決於現在，不取決於過去：

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)$$

### 2.3 價值函數

**狀態價值函數 $V^\pi(s)$**：在狀態 $s$、遵循策略 $\pi$ 的期望回報

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s \right]$$

**動作價值函數 $Q^\pi(s, a)$**：在狀態 $s$ 採取動作 $a$、之後遵循 $\pi$ 的期望回報

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s, a_0 = a \right]$$

**貝爾曼方程**（Bellman Equation）：

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a) [r + \gamma V^\pi(s')]$$

$$Q^\pi(s, a) = \sum_{s',r} P(s',r|s,a) [r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]$$

**最優貝爾曼方程**：

$$V^*(s) = \max_a \sum_{s',r} P(s',r|s,a) [r + \gamma V^*(s')]$$

$$Q^*(s, a) = \sum_{s',r} P(s',r|s,a) [r + \gamma \max_{a'} Q^*(s', a')]$$

```python
import numpy as np

class GridWorld:
    """簡單的 GridWorld MDP"""
    
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # 上、右、下、左
        # 終態（目標）
        self.terminal = [0, size * size - 1]  # 左上和右下
    
    def step(self, state, action):
        """執行動作"""
        if state in self.terminal:
            return state, 0, True
        
        row, col = state // self.size, state % self.size
        if action == 0:  # 上
            row = max(0, row - 1)
        elif action == 1:  # 右
            col = min(self.size - 1, col + 1)
        elif action == 2:  # 下
            row = min(self.size - 1, row + 1)
        elif action == 3:  # 左
            col = max(0, col - 1)
        
        next_state = row * self.size + col
        reward = 1 if next_state in self.terminal else -0.01
        done = next_state in self.terminal
        return next_state, reward, done
```

---

## 3. 動態規劃 | Dynamic Programming

### 3.1 策略迭代 | Policy Iteration

```python
def policy_iteration(env, gamma=0.9, theta=1e-6):
    """策略迭代算法"""
    n_states = env.n_states
    n_actions = env.n_actions
    
    # 隨機初始化策略和價值
    policy = np.random.randint(0, n_actions, n_states)
    V = np.zeros(n_states)
    
    while True:
        # 1. 策略評估
        while True:
            delta = 0
            for s in range(n_states):
                if s in env.terminal:
                    continue
                v = V[s]
                a = policy[s]
                next_s, r, _ = env.step(s, a)
                V[s] = r + gamma * V[next_s]
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        
        # 2. 策略改進
        policy_stable = True
        for s in range(n_states):
            if s in env.terminal:
                continue
            old_action = policy[s]
            # 選擇最優動作
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                next_s, r, _ = env.step(s, a)
                action_values[a] = r + gamma * V[next_s]
            policy[s] = np.argmax(action_values)
            
            if old_action != policy[s]:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy, V
```

### 3.2 價值迭代 | Value Iteration

```python
def value_iteration(env, gamma=0.9, theta=1e-6):
    """價值迭代算法"""
    n_states = env.n_states
    n_actions = env.n_actions
    V = np.zeros(n_states)
    
    while True:
        delta = 0
        for s in range(n_states):
            if s in env.terminal:
                continue
            v = V[s]
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                next_s, r, _ = env.step(s, a)
                action_values[a] = r + gamma * V[next_s]
            V[s] = np.max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    
    # 提取策略
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        if s in env.terminal:
            continue
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            next_s, r, _ = env.step(s, a)
            action_values[a] = r + gamma * V[next_s]
        policy[s] = np.argmax(action_values)
    
    return policy, V
```

---

## 4. 免模型算法 | Model-Free Methods

### 4.1 Monte Carlo (MC)

通過完整回合的樣本回報估計價值：

$$V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_i(s)$$

```python
def mc_prediction(env, policy, num_episodes=1000, gamma=0.9):
    """Monte Carlo 預測"""
    n_states = env.n_states
    returns = {s: [] for s in range(n_states)}
    V = np.zeros(n_states)
    
    for _ in range(num_episodes):
        episode = []
        state = np.random.randint(0, n_states)
        done = False
        
        # 生成回合
        while not done:
            action = policy[state]
            next_state, reward, done = env.step(state, action)
            episode.append((state, action, reward))
            state = next_state
        
        # 計算回報並更新價值
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G
            if s not in visited:  # First-visit MC
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                visited.add(s)
    
    return V
```

### 4.2 時序差分 (TD) 學習

結合 MC 與 DP：

$$V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

其中 $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ 是 **TD 誤差**。

```python
def td_learning(env, policy, num_episodes=1000, alpha=0.1, gamma=0.9):
    """TD(0) 學習"""
    n_states = env.n_states
    V = np.zeros(n_states)
    
    for _ in range(num_episodes):
        state = np.random.randint(0, n_states)
        done = False
        
        while not done:
            action = policy[state]
            next_state, reward, done = env.step(state, action)
            
            # TD 更新
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
    
    return V
```

### 4.3 Q-Learning

離策略（off-policy）算法，直接學最優 $Q$：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)]$$

```python
def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.9, 
               epsilon=0.1):
    """Q-Learning 算法"""
    n_states = env.n_states
    n_actions = env.n_actions
    Q = np.zeros((n_states, n_actions))
    
    for episode in range(num_episodes):
        state = np.random.randint(0, n_states)
        done = False
        
        while not done:
            # ε-greedy 策略
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done = env.step(state, action)
            
            # Q-Learning 更新
            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            
            state = next_state
    
    policy = np.argmax(Q, axis=1)
    return Q, policy
```

### 4.4 SARSA

在策略（on-policy）算法：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

---

## 5. 深度強化學習 | Deep RL

### 5.1 DQN (Deep Q-Network)

用神經網路近似 $Q(s, a; \theta)$，引入兩個關鍵技巧：

1. **經驗回放 (Experience Replay)**：打破樣本相關性
2. **目標網路 (Target Network)**：穩定訓練

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """經驗回放緩衝區"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN 智能體"""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        
        self.steps = 0
    
    def select_action(self, state):
        """ε-greedy 動作選擇"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()
    
    def train(self):
        """一步訓練"""
        if len(self.buffer) < self.batch_size:
            return
        
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # 當前 Q 值
        q_values = self.policy_net(states).gather(1, actions).squeeze()
        
        # 目標 Q 值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)
        
        # 損失
        loss = nn.functional.mse_loss(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目標網路
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # ε 衰減
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
```

### 5.2 Policy Gradient (REINFORCE)

直接優化策略：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t r_t \right]$$

策略梯度定理：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

```python
class PolicyNetwork(nn.Module):
    """策略網路"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def reinforce(env, policy_net, num_episodes=1000, gamma=0.99, lr=1e-3):
    """REINFORCE 算法"""
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    rewards_history = []
    
    for episode in range(num_episodes):
        # 生成回合
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        
        while not done:
            state_t = torch.FloatTensor(state)
            probs = policy_net(state_t)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
            
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        
        # 計算回報
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # 標準化
        
        # 策略梯度更新
        policy_loss = []
        for t in range(len(states)):
            state_t = torch.FloatTensor(states[t])
            probs = policy_net(state_t)
            action_dist = torch.distributions.Categorical(probs)
            log_prob = action_dist.log_prob(torch.tensor(actions[t]))
            policy_loss.append(-log_prob * returns[t])
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()
        
        rewards_history.append(sum(rewards))
    
    return rewards_history
```

### 5.3 Actor-Critic 架構

結合 Value-based 和 Policy-based：

- **Actor (策略網路)**：選擇動作
- **Critic (價值網路)**：評估狀態/動作

```python
class ActorCritic(nn.Module):
    """Actor-Critic 網路"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # 共享特徵層
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        # Critic
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class A2CAgent:
    """A2C (Advantage Actor-Critic) 智能體"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.gamma = gamma
        self.ac = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
    
    def select_action(self, state):
        state_t = torch.FloatTensor(state)
        probs, _ = self.ac(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update(self, log_probs, values, rewards, dones):
        """A2C 更新"""
        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns)
        
        advantages = returns - torch.cat(values).detach()
        
        actor_loss = []
        critic_loss = []
        for log_prob, value, advantage in zip(log_probs, values, advantages):
            actor_loss.append(-log_prob * advantage)
            critic_loss.append(nn.functional.mse_loss(value, torch.tensor([returns[log_probs.index(log_prob)]])))
        
        self.optimizer.zero_grad()
        loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
        loss.backward()
        self.optimizer.step()
```

### 5.4 PPO (Proximal Policy Optimization)

最流行的 on-policy 算法，限制策略更新幅度：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是重要性採樣比。

```python
class PPO:
    """Proximal Policy Optimization"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_eps=0.2, K_epochs=4, value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """PPO 更新（多次迭代同一批數據）"""
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)
        returns = torch.stack(returns)
        advantages = torch.stack(advantages)
        
        for _ in range(self.K_epochs):
            probs, values = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 重要性採樣比
            ratio = (new_log_probs - old_log_probs).exp()
            
            # PPO 裁剪目標
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 價值損失
            value_loss = nn.functional.mse_loss(values.squeeze(), returns)
            
            # 總損失
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
```

### 5.5 SAC (Soft Actor-Critic)

最大熵強化學習，平衡探索與利用：

$$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

### 5.6 算法對比

| 算法 | 類型 | 動作空間 | 樣本效率 | 穩定性 |
|------|------|----------|----------|--------|
| **DQN** | Value | 離散 | 中 | 中 |
| **REINFORCE** | Policy | 連續/離散 | 低 | 低 |
| **A2C/A3C** | Actor-Critic | 連續/離散 | 中 | 中 |
| **PPO** | Actor-Critic | 連續/離散 | 中 | 高 |
| **SAC** | Actor-Critic | 連續 | 高 | 高 |
| **TD3** | Actor-Critic | 連續 | 高 | 高 |
| **DDPG** | Actor-Critic | 連續 | 中 | 中 |

---

## 6. RLHF：LLM 對齊核心 | RLHF for LLMs

強化學習從人類反饋 (RLHF) 是 ChatGPT 等 LLM 對齊的核心技術。

### 6.1 三階段流程

```
┌──────────────────────────────────────────────────────────────┐
│  Step 1: 監督微調 SFT                                        │
│  人工編寫 (prompt, response) 對，微調 LLM                     │
│  → 得到 SFT 模型                                             │
└─────────────────┬────────────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 2: 訓練獎勵模型 RM                                      │
│  對同 prompt 採樣多個回覆，人工排序                            │
│  → 訓練 RM 學習 "好回覆" 的特徵                              │
└─────────────────┬────────────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 3: PPO 強化學習                                        │
│  用 RM 作為獎勵函數，PPO 優化策略                             │
│  → 得到對齊人類偏好的 LLM                                    │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 獎勵模型

```python
class RewardModel(nn.Module):
    """獎勵模型"""
    
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        hidden_size = base_model.config.hidden_size
        # 標量獎勵頭
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        # 取最後一個非填充 token 的隱藏狀態
        last_idx = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0)), last_idx]
        reward = self.reward_head(pooled).squeeze(-1)
        return reward

# 訓練：對 (chosen, rejected) 對，最大化 chosen 的獎勵
def train_reward_model(model, chosen_inputs, rejected_inputs, optimizer):
    chosen_rewards = model(**chosen_inputs)
    rejected_rewards = model(**rejected_inputs)
    
    # Bradley-Terry 損失
    loss = -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

### 6.3 PPO for LLM

```python
# 使用 TRL 庫
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
    gamma=0.99,
    lam=0.95,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=sft_model,
    ref_model=ref_model,         # 凍結的參考模型
    tokenizer=tokenizer,
    reward_model=reward_model,
    value_model=value_model,
)

# 訓練循環
for batch in dataloader:
    # 1. 用當前策略生成回覆
    response_tensors = ppo_trainer.generate(batch["query"])
    
    # 2. 計算獎勵
    texts = [tokenizer.decode(r) for r in response_tensors]
    rewards = [reward_model(text) for text in texts]
    
    # 3. PPO 更新
    stats = ppo_trainer.step(batch["query"], response_tensors, rewards)
```

### 6.4 DPO：簡化對齊

DPO 直接從偏好數據學習，省去 RM 和 PPO：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]$$

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=1e-6,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
)

dpo_trainer = DPOTrainer(
    model=sft_model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
```

---

## 7. 經典環境 | Classic Environments

| 環境 | 類型 | 狀態空間 | 動作空間 |
|------|------|----------|----------|
| **CartPole** | 控制 | 4 (連續) | 2 (離散) |
| **MountainCar** | 控制 | 2 (連續) | 3 (離散) |
| **Atari** | 遊戲 | 圖像 (210×160×3) | 離散 (18) |
| **MuJoCo** | 連續控制 | 物理特徵 | 連續 |
| **Go** | 棋盤 | 棋盤狀態 | 離散 (361) |
| **StarCraft II** | 策略遊戲 | 大量特徵 | 大量動作 |

```python
import gymnasium as gym

# 創建環境
env = gym.make("CartPole-v1", render_mode="human")

# 訓練循環
state, info = env.reset()
done = False
total_reward = 0
while not done:
    action = env.action_space.sample()  # 隨機策略
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state
print(f"Total reward: {total_reward}")
env.close()
```

---

## 8. 應用領域 | Applications

| 領域 | 應用 |
|------|------|
| **遊戲** | AlphaGo、OpenAI Five、StarCraft |
| **機器人** | 機械臂控制、雙足行走 |
| **自動駕駛** | 路徑規劃、決策 |
| **推薦系統** | 動態推薦、長期用戶滿意度 |
| **金融** | 算法交易、投資組合 |
| **LLM 對齊** | ChatGPT、Claude |
| **能源** | 電網控制、數據中心冷卻 |
| **醫療** | 治療策略、藥物發現 |

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  RL 算法發展路線                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  經典 (1980s-2000s)                                          │
│  ├── 動態規劃 (DP)                                           │
│  ├── Monte Carlo                                             │
│  ├── TD Learning / SARSA                                     │
│  └── Q-Learning                                              │
│                                                              │
│  深度 RL (2013-)                                              │
│  ├── DQN (2013) - 經驗回放 + 目標網路                         │
│  ├── DDPG (2015) - 連續動作                                  │
│  ├── A3C (2016) - 異步                                       │
│  ├── PPO (2017) - 當前首選                                   │
│  ├── SAC (2018) - 最大熵                                    │
│  └── TD3 (2018) - 雙 Q 學習                                  │
│                                                              │
│  LLM 對齊 (2020-)                                             │
│  ├── RLHF / PPO                                             │
│  ├── DPO (2023) - 簡化對齊                                   │
│  ├── ORPO / KTO / SimPO                                     │
│  └── GRPO (DeepSeek-R1 用於推理)                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

| 任務 | 首選算法 |
|------|----------|
| 離散動作 + 樣本足夠 | DQN |
| 連續動作 | SAC, TD3, PPO |
| 樣本效率高 | SAC, Offline RL (CQL) |
| 大模型對齊 | PPO, DPO |
| 簡單控制 | A2C, PPO |

---

## 參考文獻 | References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature* (DQN).
3. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv* (PPO).
4. Haarnoja, T., et al. (2018). Soft Actor-Critic. *ICML* (SAC).
5. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS* (InstructGPT).
6. Rafailov, R., et al. (2023). Direct Preference Optimization. *NeurIPS* (DPO).

---

## ⚡ 5 分鐘快速實戰 | Quick Practice

### 1. OpenAI Gym 環境互動

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()
total = 0
for _ in range(500):
    action = env.action_space.sample()  # 隨機策略
    obs, reward, term, trunc, info = env.step(action)
    total += reward
    if term or trunc:
        break
print(f"隨機策略得分: {total}")
env.close()
```

### 2. Stable-Baselines3 訓練 PPO

```python
# pip install stable-baselines3
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# 評估
obs, _ = env.reset()
total = 0
for _ in range(500):
    action, _ = model.predict(obs)
    obs, reward, term, trunc, _ = env.step(action)
    total += reward
    if term or trunc:
        break
print(f"PPO 得分: {total}")
```

### 3. DPO 微調 LLM

```python
# pip install trl
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

config = DPOConfig(
    beta=0.1, learning_rate=1e-6, num_train_epochs=1,
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
)
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
# 需要 chosen/rejected 字段
trainer = DPOTrainer(model=sft_model, ref_model=ref_model, 
                    args=config, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
```

---

## 🧪 練習題 | Exercises

| # | 難度 | 題目 |
|---|------|------|
| 1 | ⭐ | 在 CartPole 上訓練 DQN 達到 475+ 分 |
| 2 | ⭐⭐ | 在 LunarLander 上訓練 PPO |
| 3 | ⭐⭐ | 對比 DQN / PPO / A2C 在 Pong 環境的表現 |
| 4 | ⭐⭐⭐ | 實現 DPO 對 Llama-3-8B 做 RLHF 微調 |
| 5 | ⭐⭐⭐ | 構建 Multi-Agent RL：兩個智能體對抗訓練 |
| 6 | ⭐⭐⭐⭐ | 從零實現 PPO 訓練自己的 RLHF 模型 |

---

## 🛠 主流工具庫

| 庫 | 描述 |
|----|------|
| **Stable-Baselines3** | 最流行的 RL 庫 |
| **RLlib (Ray)** | 分散式 RL |
| **Gymnasium** | 環境標準接口 |
| **CleanRL** | 單文件實現，易學 |
| **TRL** | Transformer RL（DPO/PPO for LLM） |
| **OpenAI Spinning Up** | 教學資源 |

---

*© 2026 chanxaviersy*
