"""
================================================================================
FORZA RELATIVA (FR) TRADING AGENT - NexNow LTD
================================================================================
Agente di trading che usa gli indicatori della strategia Forza Relativa:
- Currency Strength (forza delle 8 valute)
- FR Spread (differenza di forza tra base e quote)
- ATR Dashboard (distanza dalla EMA in ATR)
- AR Lines (livelli di average range)
- Seasonal Pattern (correlazione stagionale)

L'agente impara QUANDO entrare basandosi su questi indicatori.
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Verifica GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CONFIG = {
    # Trading
    'initial_balance': 10000,
    'leverage': 30,
    'spread_pips': 1.5,
    'commission_per_lot': 7.0,
    'max_position_size': 0.1,  # Max 0.1 lotti

    # FR Strategy Thresholds (da ottimizzare)
    'fr_spread_strong': 30,      # FR > 30 = segnale forte
    'fr_spread_weak': 15,        # FR tra 15-30 = segnale debole
    'atr_percentile_high': 80,   # ATR% > 80 = troppo esteso
    'atr_percentile_low': 20,    # ATR% < 20 = in range
    'seasonal_correlation_min': 0.3,  # Correlazione stagionale minima

    # PPO Hyperparameters
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.02,        # Un po' più esplorazione
    'value_coef': 0.5,
    'max_grad_norm': 0.5,
    'learning_rate': 3e-4,
    'batch_size': 256,
    'n_epochs': 4,
    'n_steps': 512,
    'n_envs': 8,
    'total_timesteps': 300000,

    # Network
    'hidden_size': 256,
    'n_layers': 3,
}

# ============================================================================
# INDICATORI FR - ESTRAZIONE
# ============================================================================

def extract_fr_features(features_df: pd.DataFrame) -> np.ndarray:
    """
    Estrae solo le feature rilevanti per la strategia FR.

    Features estratte:
    1. fr_spread - Differenza di forza tra base e quote
    2. strength_base - Forza della valuta base
    3. strength_quote - Forza della valuta quote
    4. ema_distance_atr - Distanza dalla EMA in ATR
    5. atr_percentile - Percentile storico della distanza
    6. excess_direction - Direzione dell'eccesso (sopra/sotto EMA)
    7. dist_to_adr_high_pct - Distanza % dal livello ADR alto
    8. dist_to_adr_low_pct - Distanza % dal livello ADR basso
    9. seasonal_correlation - Correlazione con pattern stagionali
    10. seasonal_direction - Direzione prevista dal pattern
    11. rsi_14 - RSI per conferma
    12. macd_hist - MACD histogram per momentum
    13. bb_position - Posizione nelle Bollinger Bands
    14. volatility_ratio - Rapporto volatilità breve/lungo
    15. hour - Ora del giorno (sessioni)
    16. day_of_week - Giorno della settimana
    """

    fr_columns = [
        'fr_spread',
        'fr_spread_abs',
        'ema_distance_atr',
        'atr_percentile',
        'excess_direction',
        'dist_to_adr_high_pct',
        'dist_to_adr_low_pct',
        'seasonal_correlation',
        'seasonal_direction',
        'rsi_14',
        'macd_hist',
        'bb_position',
        'volatility_ratio',
        'hour',
        'day_of_week',
        'is_london_session',
        'is_ny_session',
        'is_overlap',
    ]

    # Aggiungi strength delle valute (nomi dinamici)
    strength_cols = [c for c in features_df.columns if c.startswith('strength_')]
    fr_columns.extend(strength_cols[:2])  # Prendi solo base e quote

    # Filtra colonne esistenti
    available_cols = [c for c in fr_columns if c in features_df.columns]

    # Estrai
    data = features_df[available_cols].values

    # Gestisci NaN e infiniti
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return data.astype(np.float32), available_cols


# ============================================================================
# TRADING ENVIRONMENT - FR STRATEGY
# ============================================================================

class FRTradingEnv:
    """
    Environment di trading che usa gli indicatori FR per le decisioni.

    Actions:
    - 0: HOLD (non fare nulla)
    - 1: BUY (entra long)
    - 2: SELL (entra short)
    - 3: CLOSE (chiudi posizione)

    Reward Shaping:
    - Reward base: PnL del trade
    - Bonus: Trade in direzione del FR Spread
    - Penalità: Trade contro FR Spread forte
    - Bonus: Trade durante sessioni attive
    - Penalità: Overtrading
    """

    def __init__(self, features: np.ndarray, prices: np.ndarray,
                 feature_names: list, config: dict):
        self.features = features
        self.prices = prices
        self.feature_names = feature_names
        self.config = config

        # Indici delle feature chiave
        self.fr_spread_idx = feature_names.index('fr_spread') if 'fr_spread' in feature_names else -1
        self.atr_pct_idx = feature_names.index('atr_percentile') if 'atr_percentile' in feature_names else -1
        self.session_idx = feature_names.index('is_overlap') if 'is_overlap' in feature_names else -1

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 100  # Skip warmup period
        self.balance = self.config['initial_balance']
        self.position = 0  # -1, 0, +1
        self.position_price = 0
        self.position_size = 0
        self.trades = []
        self.last_trade_step = 0

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """
        Osservazione = Features FR + stato posizione
        """
        # Features FR
        fr_features = self.features[self.current_step]

        # Stato posizione normalizzato
        position_state = np.array([
            self.position,  # -1, 0, +1
            (self.prices[self.current_step] - self.position_price) / self.position_price if self.position != 0 else 0,  # PnL%
            min((self.current_step - self.last_trade_step) / 100, 1.0),  # Tempo dall'ultimo trade
        ], dtype=np.float32)

        obs = np.concatenate([fr_features, position_state])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_fr_spread(self) -> float:
        """Ritorna il FR Spread corrente"""
        if self.fr_spread_idx >= 0:
            return self.features[self.current_step, self.fr_spread_idx]
        return 0.0

    def _is_active_session(self) -> bool:
        """Verifica se siamo in una sessione attiva"""
        if self.session_idx >= 0:
            return self.features[self.current_step, self.session_idx] > 0
        return True

    def _calculate_pnl(self, entry_price: float, exit_price: float,
                       direction: int, size: float) -> float:
        """Calcola PnL con spread e commissioni"""
        pip_value = 0.0001 if 'JPY' not in str(self.config.get('pair', '')) else 0.01
        spread_cost = self.config['spread_pips'] * pip_value * size * 100000
        commission = self.config['commission_per_lot'] * size

        if direction == 1:  # Long
            pnl = (exit_price - entry_price) * size * 100000
        else:  # Short
            pnl = (entry_price - exit_price) * size * 100000

        return pnl - spread_cost - commission

    def step(self, action: int):
        """
        Esegue un'azione e ritorna (obs, reward, done, info)
        """
        current_price = self.prices[self.current_step]
        fr_spread = self._get_fr_spread()
        is_active = self._is_active_session()

        reward = 0.0
        info = {'action': action, 'fr_spread': fr_spread}

        # === ESECUZIONE AZIONE ===

        if action == 1 and self.position == 0:  # BUY
            self.position = 1
            self.position_price = current_price
            self.position_size = self.config['max_position_size']
            self.last_trade_step = self.current_step

            # Reward shaping per direzione FR
            if fr_spread > self.config['fr_spread_strong']:
                reward += 0.1  # Bonus: FR forte in direzione del trade
            elif fr_spread < -self.config['fr_spread_weak']:
                reward -= 0.2  # Penalità: Trade contro FR

            # Bonus sessione attiva
            if is_active:
                reward += 0.05

        elif action == 2 and self.position == 0:  # SELL
            self.position = -1
            self.position_price = current_price
            self.position_size = self.config['max_position_size']
            self.last_trade_step = self.current_step

            # Reward shaping per direzione FR
            if fr_spread < -self.config['fr_spread_strong']:
                reward += 0.1  # Bonus: FR forte in direzione del trade
            elif fr_spread > self.config['fr_spread_weak']:
                reward -= 0.2  # Penalità: Trade contro FR

            # Bonus sessione attiva
            if is_active:
                reward += 0.05

        elif action == 3 and self.position != 0:  # CLOSE
            pnl = self._calculate_pnl(
                self.position_price, current_price,
                self.position, self.position_size
            )

            # Reward principale: PnL normalizzato
            reward = pnl / self.config['initial_balance'] * 10  # Scale

            self.balance += pnl
            self.trades.append({
                'entry_step': self.last_trade_step,
                'exit_step': self.current_step,
                'direction': self.position,
                'pnl': pnl,
                'fr_spread_entry': fr_spread,
            })

            self.position = 0
            self.position_price = 0
            self.position_size = 0

        elif action == 0:  # HOLD
            # Piccola penalità per tenere posizione in perdita
            if self.position != 0:
                unrealized_pnl = self._calculate_pnl(
                    self.position_price, current_price,
                    self.position, self.position_size
                )
                if unrealized_pnl < -self.config['initial_balance'] * 0.01:  # > 1% loss
                    reward -= 0.01

        # Penalità overtrading
        trades_last_24h = sum(1 for t in self.trades
                            if self.current_step - t['exit_step'] < 24)
        if trades_last_24h > 5:
            reward -= 0.05 * (trades_last_24h - 5)

        # Avanza al prossimo step
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1

        # Chiudi posizione a fine episodio
        if done and self.position != 0:
            pnl = self._calculate_pnl(
                self.position_price, self.prices[self.current_step],
                self.position, self.position_size
            )
            self.balance += pnl
            reward += pnl / self.config['initial_balance'] * 10

        obs = self._get_obs()

        return obs, reward, done, info

    @property
    def obs_dim(self) -> int:
        return len(self.feature_names) + 3  # Features + position state

    @property
    def action_dim(self) -> int:
        return 4  # HOLD, BUY, SELL, CLOSE


# ============================================================================
# VECTORIZED ENVIRONMENT
# ============================================================================

class VectorizedFREnv:
    """Multiple ambienti in parallelo per training più veloce"""

    def __init__(self, features_list: list, prices_list: list,
                 feature_names: list, config: dict, n_envs: int = 8):
        self.n_envs = n_envs
        self.envs = []

        for i in range(n_envs):
            idx = i % len(features_list)
            env = FRTradingEnv(features_list[idx], prices_list[idx],
                              feature_names, config)
            self.envs.append(env)

        self.obs_dim = self.envs[0].obs_dim
        self.action_dim = self.envs[0].action_dim

    def reset(self):
        obs = np.array([env.reset(seed=i) for i, env in enumerate(self.envs)])
        return obs

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]

        obs = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]

        # Reset degli env completati
        for i, done in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()

        return obs, rewards, dones, infos


# ============================================================================
# ACTOR-CRITIC NETWORK
# ============================================================================

class FRActorCritic(nn.Module):
    """
    Network Actor-Critic per la strategia FR.
    Include attention sui gruppi di feature FR.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_size: int = 256, n_layers: int = 3):
        super().__init__()

        # Feature encoder
        layers = [nn.Linear(obs_dim, hidden_size), nn.ReLU(), nn.LayerNorm(hidden_size)]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1),
            ])

        self.encoder = nn.Sequential(*layers)

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Inizializzazione
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(self, obs):
        features = self.encoder(obs)
        return features

    def get_action_and_value(self, obs, action=None):
        features = self.forward(obs)

        logits = self.actor(features)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), self.critic(features)

    def get_value(self, obs):
        features = self.forward(obs)
        return self.critic(features)


# ============================================================================
# PPO TRAINER
# ============================================================================

class FRPPOTrainer:
    """PPO Trainer per l'agente FR"""

    def __init__(self, env: VectorizedFREnv, config: dict):
        self.env = env
        self.config = config

        self.model = FRActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_size=config['hidden_size'],
            n_layers=config['n_layers'],
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])

        # Logging
        self.episode_rewards = []
        self.episode_lengths = []

    def collect_rollout(self, n_steps: int):
        """Raccoglie esperienza dagli ambienti"""
        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []
        logp_buf = []
        val_buf = []

        obs = torch.FloatTensor(self.env.reset()).to(device)

        for _ in range(n_steps):
            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(obs)

            obs_buf.append(obs.cpu().numpy())
            act_buf.append(action.cpu().numpy())
            logp_buf.append(log_prob.cpu().numpy())
            val_buf.append(value.squeeze(-1).cpu().numpy())

            # Step environment
            next_obs, rewards, dones, infos = self.env.step(action.cpu().numpy())

            rew_buf.append(rewards)
            done_buf.append(dones)

            obs = torch.FloatTensor(next_obs).to(device)

        # Last value for GAE
        with torch.no_grad():
            last_value = self.model.get_value(obs).squeeze(-1).cpu().numpy()

        return {
            'obs': np.array(obs_buf),
            'actions': np.array(act_buf),
            'rewards': np.array(rew_buf),
            'dones': np.array(done_buf),
            'log_probs': np.array(logp_buf),
            'values': np.array(val_buf),
            'last_value': last_value,
        }

    def compute_gae(self, rollout: dict):
        """Calcola Generalized Advantage Estimation"""
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']
        last_value = rollout['last_value']

        n_steps = len(rewards)
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.config['gamma'] * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.config['gamma'] * self.config['gae_lambda'] * next_non_terminal * last_gae

        returns = advantages + values

        return advantages, returns

    def update(self, rollout: dict, advantages: np.ndarray, returns: np.ndarray):
        """Aggiorna la policy con PPO"""
        # Flatten
        obs = rollout['obs'].reshape(-1, self.env.obs_dim)
        actions = rollout['actions'].flatten()
        old_log_probs = rollout['log_probs'].flatten()
        advantages_flat = advantages.flatten()
        returns_flat = returns.flatten()

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # Convert to tensors
        obs_t = torch.FloatTensor(obs).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(device)
        advantages_t = torch.FloatTensor(advantages_flat).to(device)
        returns_t = torch.FloatTensor(returns_flat).to(device)

        # Mini-batch updates
        batch_size = self.config['batch_size']
        n_samples = len(obs)

        total_loss = 0
        n_updates = 0

        for _ in range(self.config['n_epochs']):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                _, new_log_prob, entropy, new_value = self.model.get_action_and_value(
                    obs_t[batch_idx], actions_t[batch_idx]
                )

                # Policy loss (PPO clipping)
                ratio = torch.exp(new_log_prob - old_log_probs_t[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'],
                                   1 + self.config['clip_epsilon']) * advantages_t[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((new_value.squeeze() - returns_t[batch_idx]) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                       self.config['value_coef'] * value_loss +
                       self.config['entropy_coef'] * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return total_loss / n_updates if n_updates > 0 else 0

    def train(self, total_timesteps: int):
        """Training loop principale"""
        n_steps = self.config['n_steps']
        n_envs = self.config['n_envs']

        n_updates = total_timesteps // (n_steps * n_envs)

        print(f"\nTraining FR Agent...")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Updates: {n_updates}")
        print(f"Steps per update: {n_steps * n_envs}")
        print()

        best_reward = float('-inf')

        for update in tqdm(range(n_updates), desc="Training"):
            # Collect rollout
            rollout = self.collect_rollout(n_steps)

            # Compute GAE
            advantages, returns = self.compute_gae(rollout)

            # Update policy
            loss = self.update(rollout, advantages, returns)

            # Logging
            mean_reward = rollout['rewards'].sum(axis=0).mean()

            if update % 10 == 0:
                tqdm.write(f"Update {update}/{n_updates} | Loss: {loss:.4f} | Mean Reward: {mean_reward:.4f}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                self.save_model('best')

        self.save_model('final')
        return best_reward

    def save_model(self, suffix: str = ''):
        """Salva il modello"""
        save_dir = Path('models/fr_agent')
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f'fr_agent_{suffix}.pt' if suffix else 'fr_agent.pt'

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, save_dir / filename)

    def load_model(self, path: str):
        """Carica il modello"""
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# ============================================================================
# MAIN
# ============================================================================

def load_data(processed_dir: str = './data/processed'):
    """Carica i dati processati con feature FR"""
    processed_dir = Path(processed_dir)

    features_list = []
    prices_list = []
    feature_names = None

    pairs = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY"  # Principali per training
    ]

    print("Caricamento dati...")

    for pair in pairs:
        features_path = processed_dir / f"{pair}_features.parquet"
        raw_path = processed_dir.parent / 'raw' / f"{pair}_H1.parquet"

        if not features_path.exists():
            print(f"  {pair}: features non trovate")
            continue

        if not raw_path.exists():
            print(f"  {pair}: dati raw non trovati")
            continue

        features_df = pd.read_parquet(features_path)
        raw_df = pd.read_parquet(raw_path)

        # Estrai feature FR
        fr_features, names = extract_fr_features(features_df)

        if feature_names is None:
            feature_names = names

        # Allinea prezzi
        prices = raw_df.loc[features_df.index, 'Close'].values

        # Skip if too short
        if len(fr_features) < 1000:
            continue

        features_list.append(fr_features)
        prices_list.append(prices)

        print(f"  {pair}: {len(fr_features):,} bars, {len(names)} features")

    print(f"\nTotale: {len(features_list)} pairs caricati")
    print(f"Feature FR: {feature_names}")

    return features_list, prices_list, feature_names


def main():
    print("=" * 60)
    print("FORZA RELATIVA (FR) TRADING AGENT")
    print("=" * 60)

    # Carica dati
    features_list, prices_list, feature_names = load_data()

    if len(features_list) == 0:
        print("Errore: nessun dato caricato!")
        return

    # Crea environment
    print("\nCreazione environment...")
    env = VectorizedFREnv(
        features_list=features_list,
        prices_list=prices_list,
        feature_names=feature_names,
        config=CONFIG,
        n_envs=CONFIG['n_envs'],
    )

    print(f"Observation dim: {env.obs_dim}")
    print(f"Action dim: {env.action_dim}")

    # Crea trainer
    trainer = FRPPOTrainer(env, CONFIG)

    print(f"\nModel parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # Training
    start_time = datetime.now()
    best_reward = trainer.train(CONFIG['total_timesteps'])
    training_time = datetime.now() - start_time

    # Salva risultati
    results = {
        'training_time': str(training_time),
        'total_timesteps': CONFIG['total_timesteps'],
        'best_reward': float(best_reward),
        'config': CONFIG,
        'feature_names': feature_names,
    }

    results_path = Path('models/fr_agent/results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETATO")
    print("=" * 60)
    print(f"Tempo: {training_time}")
    print(f"Best Reward: {best_reward:.4f}")
    print(f"Modello salvato in: models/fr_agent/")


if __name__ == "__main__":
    main()
