"""
PPO Trading Agent - Reinforcement Learning per decisioni di trading
NexNow LTD - Forex AI Trading System

Il PPO agent impara:
- Quando aprire posizioni (timing)
- Quando chiudere (take profit / stop loss dinamici)
- Position sizing ottimale
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
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurazione
CONFIG = {
    # Environment
    'initial_balance': 10000,
    'leverage': 30,
    'spread_pips': 1.5,
    'commission_per_lot': 7,  # USD per lotto
    'max_position_size': 0.1,  # 10% del capitale per trade

    # PPO Hyperparameters
    'gamma': 0.99,              # Discount factor
    'gae_lambda': 0.95,         # GAE lambda
    'clip_epsilon': 0.2,        # PPO clip
    'entropy_coef': 0.01,       # Entropy bonus
    'value_coef': 0.5,          # Value loss coefficient
    'max_grad_norm': 0.5,       # Gradient clipping

    # Training - MASSIMO UTILIZZO GPU RTX 4060
    'learning_rate': 3e-4,
    'batch_size': 2048,         # Batch GRANDE per saturare GPU
    'n_epochs': 4,              # Epoche per update
    'n_steps': 1024,            # Steps prima di update
    'n_envs': 32,               # 32 ambienti paralleli!
    'total_timesteps': 2_000_000,

    # Network - Più grande per sfruttare GPU
    'hidden_size': 1024,
    'n_layers': 4,

    # Reward shaping
    'reward_scaling': 100,      # Scale rewards
    'win_bonus': 0.1,           # Bonus per trade vincente
    'drawdown_penalty': 0.5,    # Penalità per drawdown
}


class TradingEnvironment:
    """
    Ambiente di trading per RL.

    Azioni:
    - 0: HOLD (nessuna azione)
    - 1: BUY (apri long o chiudi short)
    - 2: SELL (apri short o chiudi long)
    """

    def __init__(self, features, prices, labels=None, config=CONFIG):
        self.features = features
        self.prices = prices
        self.labels = labels  # Predizioni del modello (opzionale)
        self.config = config

        self.n_steps = len(features)
        self.n_features = features.shape[1]

        # Aggiungi features di stato del portafoglio
        self.state_size = self.n_features + 4  # + position, pnl, balance, drawdown

        self.reset()

    def reset(self):
        """Reset ambiente"""
        self.current_step = 0
        self.balance = self.config['initial_balance']
        self.initial_balance = self.balance
        self.position = 0  # -1=short, 0=flat, 1=long
        self.entry_price = 0
        self.peak_balance = self.balance
        self.trades = []
        self.equity_curve = [self.balance]

        return self._get_state()

    def _get_state(self):
        """Costruisce lo stato corrente"""
        if self.current_step >= self.n_steps:
            # Padding se fuori range
            market_features = np.zeros(self.n_features, dtype=np.float32)
        else:
            market_features = self.features[self.current_step].astype(np.float32)

        # Normalizza features di portafoglio
        portfolio_features = np.array([
            self.position,  # -1, 0, 1
            np.clip(self._get_unrealized_pnl() / self.initial_balance, -1, 1),  # Normalizzato
            np.clip((self.balance - self.initial_balance) / self.initial_balance, -1, 1),  # Profit %
            np.clip(self._get_drawdown(), 0, 1),  # Drawdown %
        ], dtype=np.float32)

        state = np.concatenate([market_features, portfolio_features])

        # Handle any remaining NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        return state.astype(np.float32)

    def _get_unrealized_pnl(self):
        """Calcola PnL non realizzato"""
        if self.position == 0 or self.current_step >= len(self.prices):
            return 0

        current_price = self.prices[self.current_step]
        if self.position == 1:  # Long
            pnl = (current_price - self.entry_price) / self.entry_price
        else:  # Short
            pnl = (self.entry_price - current_price) / self.entry_price

        return pnl * self.balance * self.config['max_position_size']

    def _get_drawdown(self):
        """Calcola drawdown corrente"""
        current_equity = self.balance + self._get_unrealized_pnl()
        self.peak_balance = max(self.peak_balance, current_equity)

        if self.peak_balance > 0:
            return (self.peak_balance - current_equity) / self.peak_balance
        return 0

    def step(self, action):
        """
        Esegue un'azione nell'ambiente.

        Returns:
            state, reward, done, info
        """
        if self.current_step >= self.n_steps - 1:
            return self._get_state(), 0, True, {}

        current_price = self.prices[self.current_step]
        reward = 0
        info = {}

        # Costo spread (in percentuale)
        spread_cost = self.config['spread_pips'] * 0.0001

        # Esegui azione
        if action == 1:  # BUY
            if self.position == -1:  # Chiudi short
                pnl = (self.entry_price - current_price) / self.entry_price
                pnl -= spread_cost  # Costo chiusura
                trade_pnl = pnl * self.balance * self.config['max_position_size']
                self.balance += trade_pnl

                self.trades.append({
                    'type': 'close_short',
                    'step': self.current_step,
                    'price': current_price,
                    'pnl': trade_pnl
                })

                reward = trade_pnl / self.initial_balance * self.config['reward_scaling']
                if trade_pnl > 0:
                    reward += self.config['win_bonus']

                self.position = 0
                self.entry_price = 0

            elif self.position == 0:  # Apri long
                self.position = 1
                self.entry_price = current_price * (1 + spread_cost)  # Include spread

                self.trades.append({
                    'type': 'open_long',
                    'step': self.current_step,
                    'price': self.entry_price,
                    'pnl': 0
                })

        elif action == 2:  # SELL
            if self.position == 1:  # Chiudi long
                pnl = (current_price - self.entry_price) / self.entry_price
                pnl -= spread_cost  # Costo chiusura
                trade_pnl = pnl * self.balance * self.config['max_position_size']
                self.balance += trade_pnl

                self.trades.append({
                    'type': 'close_long',
                    'step': self.current_step,
                    'price': current_price,
                    'pnl': trade_pnl
                })

                reward = trade_pnl / self.initial_balance * self.config['reward_scaling']
                if trade_pnl > 0:
                    reward += self.config['win_bonus']

                self.position = 0
                self.entry_price = 0

            elif self.position == 0:  # Apri short
                self.position = -1
                self.entry_price = current_price * (1 - spread_cost)  # Include spread

                self.trades.append({
                    'type': 'open_short',
                    'step': self.current_step,
                    'price': self.entry_price,
                    'pnl': 0
                })

        # Penalità drawdown
        drawdown = self._get_drawdown()
        if drawdown > 0.1:  # > 10% drawdown
            reward -= drawdown * self.config['drawdown_penalty']

        # Avanza
        self.current_step += 1
        self.equity_curve.append(self.balance + self._get_unrealized_pnl())

        # Check done
        done = self.current_step >= self.n_steps - 1

        # Chiudi posizioni a fine episodio
        if done and self.position != 0:
            final_price = self.prices[min(self.current_step, len(self.prices)-1)]
            if self.position == 1:
                pnl = (final_price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - final_price) / self.entry_price
            pnl -= spread_cost
            trade_pnl = pnl * self.balance * self.config['max_position_size']
            self.balance += trade_pnl
            reward += trade_pnl / self.initial_balance * self.config['reward_scaling']

        # Info
        info = {
            'balance': self.balance,
            'position': self.position,
            'n_trades': len([t for t in self.trades if 'close' in t['type']]),
            'drawdown': drawdown
        }

        return self._get_state(), reward, done, info

    def get_metrics(self):
        """Calcola metriche finali"""
        closed_trades = [t for t in self.trades if 'close' in t['type']]

        if not closed_trades:
            return {
                'total_return': 0,
                'n_trades': 0,
                'win_rate': 0,
                'avg_trade': 0,
                'max_drawdown': 0,
                'sharpe': 0
            }

        wins = sum(1 for t in closed_trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in closed_trades)

        # Calcola max drawdown da equity curve
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = drawdown.max() * 100

        # Sharpe (semplificato)
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)  # Annualizzato

        return {
            'total_return': (self.balance / self.initial_balance - 1) * 100,
            'n_trades': len(closed_trades),
            'win_rate': wins / len(closed_trades) * 100 if closed_trades else 0,
            'avg_trade': total_pnl / len(closed_trades) if closed_trades else 0,
            'max_drawdown': max_dd,
            'sharpe': sharpe
        }


class VectorizedTradingEnv:
    """
    Ambienti di trading vettorizzati per GPU.
    Esegue N ambienti in parallelo per saturare la GPU.
    """

    def __init__(self, all_data, n_envs, config=CONFIG):
        self.n_envs = n_envs
        self.config = config
        self.all_data = all_data  # Lista di (features, prices, labels) per ogni pair

        # Crea N ambienti
        self.envs = []
        for i in range(n_envs):
            # Seleziona dati random
            idx = i % len(all_data)
            features, prices, labels = all_data[idx]

            # Punto di partenza random
            max_start = len(features) - 2000
            if max_start > 0:
                start = np.random.randint(0, max_start)
            else:
                start = 0

            ep_len = min(2000, len(features) - start)
            env = TradingEnvironment(
                features[start:start+ep_len],
                prices[start:start+ep_len],
                labels.iloc[start:start+ep_len] if labels is not None else None,
                config
            )
            self.envs.append(env)

        self.state_size = self.envs[0].state_size

    def reset(self):
        """Reset tutti gli ambienti"""
        states = []
        for i, env in enumerate(self.envs):
            # Riassegna dati random
            idx = np.random.randint(0, len(self.all_data))
            features, prices, labels = self.all_data[idx]

            max_start = len(features) - 2000
            start = np.random.randint(0, max(1, max_start))
            ep_len = min(2000, len(features) - start)

            self.envs[i] = TradingEnvironment(
                features[start:start+ep_len],
                prices[start:start+ep_len],
                labels.iloc[start:start+ep_len] if labels is not None else None,
                self.config
            )
            states.append(self.envs[i].reset())

        return np.array(states, dtype=np.float32)

    def step(self, actions):
        """Step tutti gli ambienti in parallelo"""
        states = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, done, info = env.step(action)

            if done:
                # Auto-reset
                idx = np.random.randint(0, len(self.all_data))
                features, prices, labels = self.all_data[idx]
                max_start = len(features) - 2000
                start = np.random.randint(0, max(1, max_start))
                ep_len = min(2000, len(features) - start)

                self.envs[i] = TradingEnvironment(
                    features[start:start+ep_len],
                    prices[start:start+ep_len],
                    labels.iloc[start:start+ep_len] if labels is not None else None,
                    self.config
                )
                state = self.envs[i].reset()

            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.array(states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones),
            infos
        )

    def get_metrics(self):
        """Metriche aggregate"""
        all_metrics = [env.get_metrics() for env in self.envs]
        return {
            'avg_return': np.mean([m['total_return'] for m in all_metrics]),
            'avg_win_rate': np.mean([m['win_rate'] for m in all_metrics]),
            'avg_trades': np.mean([m['n_trades'] for m in all_metrics]),
            'avg_sharpe': np.mean([m['sharpe'] for m in all_metrics])
        }


class ActorCritic(nn.Module):
    """
    Rete Actor-Critic per PPO.

    Actor: Policy network (azioni)
    Critic: Value network (stima valore stato)
    """

    def __init__(self, state_size, action_size, hidden_size=256, n_layers=2):
        super().__init__()

        # Shared feature extractor
        layers = []
        in_size = state_size
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_size = hidden_size

        self.shared = nn.Sequential(*layers)

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def get_action(self, state, deterministic=False):
        """Seleziona azione"""
        action_probs, value = self.forward(state)

        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()

        return action, action_probs, value

    def evaluate(self, states, actions):
        """Valuta azioni per training"""
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)

        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return action_log_probs, values.squeeze(-1), entropy


class PPOAgent:
    """PPO Agent per trading"""

    def __init__(self, state_size, action_size=3, config=CONFIG, device='cuda'):
        self.config = config
        self.device = device

        # Network
        self.network = ActorCritic(
            state_size, action_size,
            config['hidden_size'], config['n_layers']
        ).to(device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config['learning_rate']
        )

        # Memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state, deterministic=False):
        """Seleziona azione dato lo stato"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, action_probs, value = self.network.get_action(state_tensor, deterministic)

        if not deterministic:
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(action)
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())

        return action.item()

    def store_transition(self, state, action, reward, done):
        """Memorizza transizione"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value):
        """Calcola Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.config['gamma'] * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]

        return advantages, returns

    def update(self):
        """Aggiorna policy con PPO"""
        if len(self.states) == 0:
            return {}

        # Converti a tensori
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Calcola GAE
        with torch.no_grad():
            _, next_value = self.network(states[-1:])
            next_value = next_value.item()

        advantages, returns = self.compute_gae(next_value)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalizza advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training epochs
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0

        indices = np.arange(len(states))

        for _ in range(self.config['n_epochs']):
            np.random.shuffle(indices)

            for start in range(0, len(states), self.config['batch_size']):
                end = start + self.config['batch_size']
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate current policy
                new_log_probs, values, entropy = self.network.evaluate(batch_states, batch_actions)

                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon']) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = nn.MSELoss()(values, batch_returns)

                # Total loss
                loss = (
                    actor_loss +
                    self.config['value_coef'] * critic_loss -
                    self.config['entropy_coef'] * entropy.mean()
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss += actor_loss.item()
                value_loss += critic_loss.item()
                entropy_loss += entropy.mean().item()

        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        n_updates = self.config['n_epochs'] * (len(indices) // self.config['batch_size'] + 1)

        return {
            'total_loss': total_loss / n_updates,
            'policy_loss': policy_loss / n_updates,
            'value_loss': value_loss / n_updates,
            'entropy': entropy_loss / n_updates
        }

    def save(self, path):
        """Salva modello"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load(self, path):
        """Carica modello"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def load_training_data(data_dir):
    """Carica dati per training"""
    data_dir = Path(data_dir)

    # Carica features aggregate
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet")
    X_val = pd.read_parquet(data_dir / "X_val.parquet")
    y_val = pd.read_parquet(data_dir / "y_val.parquet")

    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Val: {len(X_val)} samples")

    return X_train, y_train, X_val, y_val


def load_pair_data(data_dir, pair):
    """Carica dati per una coppia specifica"""
    data_dir = Path(data_dir)
    raw_dir = data_dir.parent / "raw"

    # Features
    features = pd.read_parquet(data_dir / f"{pair}_features.parquet")
    labels = pd.read_parquet(data_dir / f"{pair}_labels.parquet")

    # Prezzi raw
    prices = None
    for tf in ['H1', 'H4', 'D1']:
        raw_file = raw_dir / f"{pair}_{tf}.parquet"
        if raw_file.exists():
            raw_df = pd.read_parquet(raw_file)
            raw_df.columns = [c.lower() for c in raw_df.columns]
            prices = raw_df['close'].values
            break

    if prices is None:
        raise ValueError(f"No price data found for {pair}")

    # Allinea lunghezze
    min_len = min(len(features), len(labels), len(prices))
    features = features.iloc[:min_len].values.astype(np.float32)
    labels = labels.iloc[:min_len]
    prices = prices[:min_len].astype(np.float32)

    # Handle NaN e Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalizza features (clip estremi)
    features = np.clip(features, -10, 10)

    return features, prices, labels


def train_ppo(config=CONFIG):
    """Training principale PPO con ambienti vettorizzati per GPU"""

    logger.info("=" * 60)
    logger.info("PPO TRADING AGENT - TRAINING (GPU OPTIMIZED)")
    logger.info("=" * 60)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.info("Warning: Training su CPU (sarà lento)")

    # Directories
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models" / "ppo"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Lista coppie
    pairs = sorted(set(f.stem.replace('_features', '') for f in data_dir.glob("*_features.parquet")))
    logger.info(f"Coppie disponibili: {len(pairs)}")

    # Seleziona coppie per training
    main_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURJPY', 'GBPJPY', 'EURGBP']
    train_pairs = [p for p in main_pairs if p in pairs]
    if not train_pairs:
        train_pairs = pairs[:8]

    logger.info(f"Training su: {train_pairs}")

    # Pre-carica TUTTI i dati in memoria (per velocità)
    logger.info("Caricamento dati in memoria...")
    all_data = []
    for pair in train_pairs:
        try:
            features, prices, labels = load_pair_data(data_dir, pair)
            all_data.append((features, prices, labels))
            logger.info(f"  {pair}: {len(features):,} samples")
        except Exception as e:
            logger.warning(f"  {pair}: SKIP - {e}")

    if not all_data:
        raise ValueError("Nessun dato caricato!")

    state_size = all_data[0][0].shape[1] + 4  # + portfolio features
    n_envs = config['n_envs']

    logger.info(f"\nState size: {state_size}")
    logger.info(f"Ambienti paralleli: {n_envs}")

    # Crea ambiente vettorizzato
    vec_env = VectorizedTradingEnv(all_data, n_envs, config)

    # Crea agent
    agent = PPOAgent(state_size, action_size=3, config=config, device=device)

    n_params = sum(p.numel() for p in agent.network.parameters())
    logger.info(f"Network parameters: {n_params:,}")

    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)

    best_sharpe = -np.inf
    episode_rewards = deque(maxlen=100)

    total_steps = 0
    update_count = 0

    start_time = datetime.now()

    # Reset iniziale
    states = vec_env.reset()

    # Buffer per batch
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_dones = []
    batch_values = []
    batch_log_probs = []

    while total_steps < config['total_timesteps']:
        # Batch di azioni per tutti gli ambienti
        states_tensor = torch.FloatTensor(states).to(device)

        with torch.no_grad():
            action_probs, values = agent.network(states_tensor)
            dist = Categorical(action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        actions_np = actions.cpu().numpy()

        # Step tutti gli ambienti
        next_states, rewards, dones, infos = vec_env.step(actions_np)

        # Accumula nel buffer
        batch_states.append(states)
        batch_actions.append(actions_np)
        batch_rewards.append(rewards)
        batch_dones.append(dones)
        batch_values.append(values.cpu().numpy().flatten())
        batch_log_probs.append(log_probs.cpu().numpy())

        states = next_states
        total_steps += n_envs

        # Track rewards per episodio completato
        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(sum(batch_rewards[-1]))

        # Update ogni n_steps * n_envs
        if len(batch_states) >= config['n_steps']:
            # Prepara tensori
            all_states = np.concatenate(batch_states, axis=0)
            all_actions = np.concatenate(batch_actions, axis=0)
            all_rewards = np.concatenate(batch_rewards, axis=0)
            all_dones = np.concatenate(batch_dones, axis=0)
            all_values = np.concatenate(batch_values, axis=0)
            all_log_probs = np.concatenate(batch_log_probs, axis=0)

            # Calcola GAE
            with torch.no_grad():
                next_states_tensor = torch.FloatTensor(states).to(device)
                _, next_values = agent.network(next_states_tensor)
                next_values = next_values.cpu().numpy().flatten()

            # GAE per ogni ambiente
            advantages = np.zeros_like(all_rewards)
            returns = np.zeros_like(all_rewards)

            for t in reversed(range(len(all_rewards))):
                if t == len(all_rewards) - 1:
                    next_val = next_values[t % n_envs]
                else:
                    next_val = all_values[t + 1]

                delta = all_rewards[t] + config['gamma'] * next_val * (1 - all_dones[t]) - all_values[t]

                if t == len(all_rewards) - 1:
                    advantages[t] = delta
                else:
                    advantages[t] = delta + config['gamma'] * config['gae_lambda'] * (1 - all_dones[t]) * advantages[t + 1]

                returns[t] = advantages[t] + all_values[t]

            # Normalizza advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update
            states_t = torch.FloatTensor(all_states).to(device)
            actions_t = torch.LongTensor(all_actions).to(device)
            old_log_probs_t = torch.FloatTensor(all_log_probs).to(device)
            advantages_t = torch.FloatTensor(advantages).to(device)
            returns_t = torch.FloatTensor(returns).to(device)

            # Mini-batch updates
            indices = np.arange(len(all_states))

            for _ in range(config['n_epochs']):
                np.random.shuffle(indices)

                for start in range(0, len(indices), config['batch_size']):
                    end = start + config['batch_size']
                    mb_indices = indices[start:end]

                    mb_states = states_t[mb_indices]
                    mb_actions = actions_t[mb_indices]
                    mb_old_log_probs = old_log_probs_t[mb_indices]
                    mb_advantages = advantages_t[mb_indices]
                    mb_returns = returns_t[mb_indices]

                    # Forward
                    new_log_probs, values, entropy = agent.network.evaluate(mb_states, mb_actions)

                    # PPO loss
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - config['clip_epsilon'], 1 + config['clip_epsilon']) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    critic_loss = nn.MSELoss()(values, mb_returns)

                    loss = actor_loss + config['value_coef'] * critic_loss - config['entropy_coef'] * entropy.mean()

                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.network.parameters(), config['max_grad_norm'])
                    agent.optimizer.step()

            # Clear buffer
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_dones = []
            batch_values = []
            batch_log_probs = []

            update_count += 1

            # Log ogni 10 update
            if update_count % 10 == 0:
                metrics = vec_env.get_metrics()
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                steps_per_sec = total_steps / (datetime.now() - start_time).total_seconds()

                logger.info(
                    f"Update {update_count} | Steps: {total_steps:,} ({steps_per_sec:.0f}/s) | "
                    f"Return: {metrics['avg_return']:.2f}% | "
                    f"WinRate: {metrics['avg_win_rate']:.1f}% | "
                    f"Sharpe: {metrics['avg_sharpe']:.2f}"
                )

            # Validation ogni 50 update
            if update_count % 50 == 0:
                val_metrics = validate_agent(agent, data_dir, train_pairs[:3], config)

                logger.info(f"\n--- VALIDATION ---")
                logger.info(f"Avg Return: {val_metrics['avg_return']:.2f}%")
                logger.info(f"Avg Sharpe: {val_metrics['avg_sharpe']:.2f}")
                logger.info(f"-" * 40 + "\n")

                if val_metrics['avg_sharpe'] > best_sharpe:
                    best_sharpe = val_metrics['avg_sharpe']
                    agent.save(models_dir / "best_model.pt")
                    logger.info(f"  >> Nuovo best model! Sharpe: {best_sharpe:.2f}")

            # Checkpoint ogni 200 update
            if update_count % 200 == 0:
                agent.save(models_dir / f"checkpoint_{update_count}.pt")

    # Training completato
    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETATO!")
    logger.info("=" * 60)
    logger.info(f"Tempo totale: {elapsed}")
    logger.info(f"Updates: {update_count}")
    logger.info(f"Steps: {total_steps:,}")
    logger.info(f"Steps/sec: {total_steps / elapsed.total_seconds():.0f}")
    logger.info(f"Best Sharpe: {best_sharpe:.2f}")

    # Salva modello finale
    agent.save(models_dir / "final_model.pt")

    # Salva risultati
    results = {
        'training_time': str(elapsed),
        'total_updates': update_count,
        'total_steps': total_steps,
        'steps_per_second': total_steps / elapsed.total_seconds(),
        'best_sharpe': best_sharpe,
        'config': {k: str(v) for k, v in config.items()}
    }

    with open(models_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return agent


def validate_agent(agent, data_dir, pairs, config):
    """Valida agent su dati non visti"""

    metrics_list = []

    for pair in pairs:
        features, prices, labels = load_pair_data(data_dir, pair)

        # Usa ultimi 20% come validation
        val_start = int(len(features) * 0.8)
        val_features = features[val_start:]
        val_prices = prices[val_start:]
        val_labels = labels.iloc[val_start:]

        env = TradingEnvironment(val_features, val_prices, val_labels, config)
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            state, _, done, _ = env.step(action)

        metrics = env.get_metrics()
        metrics_list.append(metrics)

    return {
        'avg_return': np.mean([m['total_return'] for m in metrics_list]),
        'avg_sharpe': np.mean([m['sharpe'] for m in metrics_list]),
        'avg_win_rate': np.mean([m['win_rate'] for m in metrics_list]),
        'avg_trades': np.mean([m['n_trades'] for m in metrics_list])
    }


def main():
    parser = argparse.ArgumentParser(description='Train PPO Trading Agent')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Total timesteps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    args = parser.parse_args()

    config = CONFIG.copy()
    config['total_timesteps'] = args.timesteps
    config['learning_rate'] = args.lr

    train_ppo(config)


if __name__ == "__main__":
    main()
