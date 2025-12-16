"""
PPO Trading Agent v2 - Versione Stabile e Professionale
Basato su best practices di RL per trading
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from pathlib import Path
from datetime import datetime
import json
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAZIONE STABILE
# ============================================================================
CONFIG = {
    # Environment
    'initial_balance': 10000,
    'transaction_cost': 0.0002,  # 2 pips spread semplificato
    'max_position': 1.0,         # Posizione massima normalizzata

    # PPO - Valori standard stabili
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'max_grad_norm': 0.5,

    # Training
    'learning_rate': 1e-4,       # LR più basso = più stabile
    'batch_size': 256,
    'n_epochs': 4,
    'n_steps': 256,
    'n_envs': 8,                 # Meno env = più stabile
    'total_timesteps': 500_000,

    # Network - Più piccolo = più stabile
    'hidden_size': 128,
    'n_layers': 2,
}


# ============================================================================
# ENVIRONMENT SEMPLIFICATO E STABILE
# ============================================================================
class SimpleTradingEnv:
    """
    Ambiente di trading semplificato.

    Stato: features di mercato + posizione corrente
    Azioni: 0=SHORT, 1=FLAT, 2=LONG
    Reward: variazione percentuale del portafoglio
    """

    def __init__(self, prices, features):
        self.prices = prices
        self.features = features
        self.n_steps = len(prices)
        self.n_features = features.shape[1]

        # State = features + [position, returns_so_far]
        self.state_size = self.n_features + 2

        self.reset()

    def reset(self):
        self.step_idx = 0
        self.position = 0  # -1, 0, 1
        self.entry_price = 0
        self.total_return = 0
        self.returns_history = []

        return self._get_state()

    def _get_state(self):
        if self.step_idx >= self.n_steps:
            feat = np.zeros(self.n_features, dtype=np.float32)
        else:
            feat = self.features[self.step_idx].astype(np.float32)

        # Aggiungi info portafoglio
        state = np.concatenate([
            feat,
            [self.position, np.clip(self.total_return, -1, 1)]
        ])

        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, action):
        """
        action: 0=SHORT, 1=FLAT, 2=LONG
        """
        if self.step_idx >= self.n_steps - 1:
            return self._get_state(), 0, True, {}

        current_price = self.prices[self.step_idx]
        next_price = self.prices[self.step_idx + 1]

        # Converti azione in posizione target
        target_position = action - 1  # 0->-1, 1->0, 2->1

        # Calcola reward basato sulla posizione ATTUALE (non target)
        price_return = (next_price - current_price) / current_price

        # Reward = rendimento * posizione (semplice e stabile)
        reward = self.position * price_return

        # Costo transazione se cambia posizione
        if target_position != self.position:
            reward -= CONFIG['transaction_cost']

        # Aggiorna posizione
        self.position = target_position

        # Track returns
        self.total_return += reward
        self.returns_history.append(reward)

        # Avanza
        self.step_idx += 1
        done = self.step_idx >= self.n_steps - 1

        return self._get_state(), reward, done, {'return': self.total_return}

    def get_metrics(self):
        if not self.returns_history:
            return {'total_return': 0, 'sharpe': 0, 'n_trades': 0}

        returns = np.array(self.returns_history)
        total_ret = self.total_return * 100

        # Sharpe annualizzato (assumendo H1 data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        else:
            sharpe = 0

        return {
            'total_return': total_ret,
            'sharpe': sharpe,
            'n_trades': len(returns)
        }


class VecEnv:
    """Ambienti vettorizzati semplici"""

    def __init__(self, all_data, n_envs):
        self.all_data = all_data
        self.n_envs = n_envs
        self.envs = [self._make_env() for _ in range(n_envs)]
        self.state_size = self.envs[0].state_size

    def _make_env(self):
        idx = np.random.randint(len(self.all_data))
        prices, features = self.all_data[idx]

        # Finestra random
        max_start = max(0, len(prices) - 1000)
        start = np.random.randint(0, max_start + 1)
        end = min(start + 1000, len(prices))

        return SimpleTradingEnv(prices[start:end], features[start:end])

    def reset(self):
        self.envs = [self._make_env() for _ in range(self.n_envs)]
        return np.array([e.reset() for e in self.envs], dtype=np.float32)

    def step(self, actions):
        states, rewards, dones = [], [], []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            s, r, d, _ = env.step(action)

            if d:
                self.envs[i] = self._make_env()
                s = self.envs[i].reset()

            states.append(s)
            rewards.append(r)
            dones.append(d)

        return (
            np.array(states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones)
        )

    def get_avg_metrics(self):
        metrics = [e.get_metrics() for e in self.envs]
        return {
            'avg_return': np.mean([m['total_return'] for m in metrics]),
            'avg_sharpe': np.mean([m['sharpe'] for m in metrics])
        }


# ============================================================================
# NETWORK SEMPLICE E STABILE
# ============================================================================
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size=3, hidden=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.actor = nn.Linear(hidden, action_size)
        self.critic = nn.Linear(hidden, 1)

        # Inizializzazione ortogonale (più stabile)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.shared(x)
        return torch.softmax(self.actor(h), dim=-1), self.critic(h)

    def get_action(self, state):
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states, actions):
        probs, values = self.forward(states)
        dist = Categorical(probs)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy()


# ============================================================================
# PPO AGENT
# ============================================================================
class PPOAgent:
    def __init__(self, state_size, device='cuda'):
        self.device = device
        self.network = ActorCritic(
            state_size,
            hidden=CONFIG['hidden_size']
        ).to(device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=CONFIG['learning_rate'],
            eps=1e-5
        )

    def act(self, states):
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            actions, log_probs, values = self.network.get_action(states_t)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, states, actions, rewards, dones, old_log_probs, old_values):
        """PPO update con GAE"""

        # Calcola returns e advantages con GAE
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        returns = np.zeros(n_steps, dtype=np.float32)

        gae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = 0
            else:
                next_value = old_values[t + 1]

            delta = rewards[t] + CONFIG['gamma'] * next_value * (1 - dones[t]) - old_values[t]
            gae = delta + CONFIG['gamma'] * CONFIG['gae_lambda'] * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + old_values[t]

        # Normalizza advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Converti a tensori
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Mini-batch training
        n_samples = len(states)
        indices = np.arange(n_samples)

        total_loss = 0
        n_updates = 0

        for _ in range(CONFIG['n_epochs']):
            np.random.shuffle(indices)

            for start in range(0, n_samples, CONFIG['batch_size']):
                end = start + CONFIG['batch_size']
                mb_idx = indices[start:end]

                new_log_probs, values, entropy = self.network.evaluate(
                    states_t[mb_idx],
                    actions_t[mb_idx]
                )

                # PPO clipped loss
                ratio = torch.exp(new_log_probs - old_log_probs_t[mb_idx])
                surr1 = ratio * advantages_t[mb_idx]
                surr2 = torch.clamp(ratio, 1 - CONFIG['clip_epsilon'], 1 + CONFIG['clip_epsilon']) * advantages_t[mb_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (returns_t[mb_idx] - values).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = actor_loss + CONFIG['value_coef'] * critic_loss + CONFIG['entropy_coef'] * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), CONFIG['max_grad_norm'])
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return total_loss / max(n_updates, 1)

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(data_dir):
    """Carica dati in formato semplice (prices, features)"""
    data_dir = Path(data_dir)
    raw_dir = data_dir.parent / "raw"

    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    all_data = []

    for pair in pairs:
        try:
            # Carica features
            feat_file = data_dir / f"{pair}_features.parquet"
            if not feat_file.exists():
                continue

            features = pd.read_parquet(feat_file).values.astype(np.float32)

            # Carica prezzi
            for tf in ['H1', 'H4', 'D1']:
                raw_file = raw_dir / f"{pair}_{tf}.parquet"
                if raw_file.exists():
                    raw_df = pd.read_parquet(raw_file)
                    raw_df.columns = [c.lower() for c in raw_df.columns]
                    prices = raw_df['close'].values.astype(np.float32)
                    break
            else:
                continue

            # Allinea
            min_len = min(len(prices), len(features))
            prices = prices[:min_len]
            features = features[:min_len]

            # Pulisci
            features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
            features = np.clip(features, -5, 5)  # Clip più aggressivo

            all_data.append((prices, features))
            logger.info(f"  {pair}: {len(prices):,} samples")

        except Exception as e:
            logger.warning(f"  {pair}: skip - {e}")

    return all_data


# ============================================================================
# TRAINING
# ============================================================================
def train():
    logger.info("=" * 60)
    logger.info("PPO TRADING AGENT v2 - TRAINING STABILE")
    logger.info("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "processed"
    model_dir = base_dir / "models" / "ppo_v2"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("\nCaricamento dati...")
    all_data = load_data(data_dir)

    if not all_data:
        raise ValueError("Nessun dato caricato!")

    # Create env e agent
    vec_env = VecEnv(all_data, CONFIG['n_envs'])
    agent = PPOAgent(vec_env.state_size, device)

    n_params = sum(p.numel() for p in agent.network.parameters())
    logger.info(f"Network: {n_params:,} parametri")
    logger.info(f"Ambienti: {CONFIG['n_envs']}")

    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)

    states = vec_env.reset()

    # Buffers
    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_log_probs = []
    all_values = []

    total_steps = 0
    update_count = 0
    best_return = -np.inf
    recent_returns = deque(maxlen=20)

    start_time = datetime.now()

    while total_steps < CONFIG['total_timesteps']:
        # Collect rollout
        actions, log_probs, values = agent.act(states)
        next_states, rewards, dones = vec_env.step(actions)

        # Store
        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_dones.append(dones)
        all_log_probs.append(log_probs)
        all_values.append(values)

        states = next_states
        total_steps += CONFIG['n_envs']

        # Update quando abbiamo abbastanza dati
        if len(all_states) >= CONFIG['n_steps']:
            # Flatten
            batch_states = np.concatenate(all_states)
            batch_actions = np.concatenate(all_actions)
            batch_rewards = np.concatenate(all_rewards)
            batch_dones = np.concatenate(all_dones)
            batch_log_probs = np.concatenate(all_log_probs)
            batch_values = np.concatenate(all_values)

            # Update
            loss = agent.update(
                batch_states, batch_actions, batch_rewards,
                batch_dones, batch_log_probs, batch_values
            )

            # Clear
            all_states, all_actions, all_rewards = [], [], []
            all_dones, all_log_probs, all_values = [], [], []

            update_count += 1

            # Metrics
            metrics = vec_env.get_avg_metrics()
            recent_returns.append(metrics['avg_return'])
            avg_return = np.mean(recent_returns)

            # Log ogni 5 update
            if update_count % 5 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = total_steps / elapsed

                logger.info(
                    f"Update {update_count:3d} | "
                    f"Steps: {total_steps:,} ({speed:.0f}/s) | "
                    f"Return: {avg_return:+.2f}% | "
                    f"Sharpe: {metrics['avg_sharpe']:.2f} | "
                    f"Loss: {loss:.4f}"
                )

            # Save best
            if avg_return > best_return:
                best_return = avg_return
                agent.save(model_dir / "best_model.pt")

    # Salva finale
    elapsed = datetime.now() - start_time
    agent.save(model_dir / "final_model.pt")

    results = {
        'training_time': str(elapsed),
        'total_steps': total_steps,
        'best_return': float(best_return),
        'config': CONFIG
    }

    with open(model_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETATO!")
    logger.info("=" * 60)
    logger.info(f"Tempo: {elapsed}")
    logger.info(f"Best Return: {best_return:+.2f}%")
    logger.info(f"Modello: {model_dir / 'best_model.pt'}")


if __name__ == "__main__":
    train()
