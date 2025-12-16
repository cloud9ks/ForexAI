# 🤖 Forex AI Trading Agent

## NexNow LTD - Sistema di Trading Automatico con AI

---

## 📁 Struttura Progetto

```
forex_ai_project/
│
├── data_collection/
│   ├── download_forex_data.py     # Script per scaricare dati da MT5
│   └── compute_features.py        # Calcolo features e indicatori
│
├── data/
│   ├── raw/                       # Dati OHLCV grezzi
│   ├── processed/                 # Features e labels
│   └── external/                  # Dati economici, COT, etc.
│
├── models/                        # Modelli salvati
├── logs/                          # Log di training
│
├── requirements.txt               # Dipendenze Python
├── setup_and_run.py              # Script principale
└── GUIDA_AI_TRADING_AGENT.md     # Guida completa
```

---

## 🚀 Quick Start

### 1. Setup Ambiente

```bash
# Crea ambiente conda
conda create -n forex-ai python=3.10
conda activate forex-ai

# Installa PyTorch con GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Installa altre dipendenze
pip install -r requirements.txt
```

### 2. Verifica Ambiente

```bash
python setup_and_run.py --check
```

Output atteso:
```
✓ Python: 3.10.x
✓ pandas
✓ numpy
✓ torch
✓ GPU: NVIDIA GeForce RTX 4090 (24.0 GB)
✓ MT5: Connesso (Account: xxxxx)
✅ Ambiente OK!
```

### 3. Scarica Dati (richiede MT5 su Windows)

```bash
# 10 anni di storico per tutti i 28 pair
python setup_and_run.py --download --years 10
```

**Alternativa senza MT5:**
```bash
# Usa dati da Yahoo Finance (meno completi)
python -c "from setup_and_run import download_alternative_data; download_alternative_data(10)"
```

### 4. Calcola Features

```bash
python setup_and_run.py --features
```

### 5. Crea Dataset

```bash
python setup_and_run.py --dataset
```

### 6. Esegui Tutto Insieme

```bash
python setup_and_run.py --all
```

---

## 📊 Features Calcolate

Il sistema calcola automaticamente 80+ features per ogni pair:

### Indicatori Tecnici Standard
- RSI (7, 14, 21 periodi)
- MACD
- Bollinger Bands
- Stochastic
- ATR
- Moving Averages (SMA/EMA 10, 20, 50, 100, 200)

### Indicatori Custom (dal sistema Forza Relativa)
- **Currency Strength**: Forza delle 8 valute (USD, EUR, GBP, CHF, CAD, JPY, AUD, NZD)
- **FR Spread**: Differenza di forza tra base e quote
- **ATR Dashboard**: Distanza dalla EMA in multipli ATR + percentile storico
- **AR Lines**: Livelli di range medio (ADR/AWR)
- **Seasonal Pattern**: Correlazione con pattern storici

### Features Temporali
- Ora, giorno della settimana, mese
- Sessioni di trading (London, NY, overlap)

---

## 🧠 Modelli Supportati

### Opzione 1: Temporal Fusion Transformer (Consigliato)
- ~80M parametri
- ~6GB VRAM
- Ottimo per previsioni multi-orizzonte

### Opzione 2: Custom Transformer
- ~100M parametri
- ~8GB VRAM
- Più flessibile

### Opzione 3: PPO (Reinforcement Learning)
- Per decisioni di trading end-to-end
- Da usare dopo il predictor

---

## 💻 Requisiti Hardware

| Componente | Minimo | Consigliato |
|------------|--------|-------------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| RAM | 32GB | 64GB |
| Storage | 500GB SSD | 1TB NVMe |
| CPU | 8 core | 16+ core |

---

## 📈 Flusso di Training

```
1. Download Dati (MT5) ──────────────────────────────────┐
                                                          │
2. Feature Engineering ──► 80+ features per bar           │
                                                          │
3. Create Labels ──► Direction, Target Return, Win Prob   │
                                                          ▼
4. Train Predictor (TFT) ──► Predice direzione + target
                                                          │
5. Train Agent (PPO) ──► Decide quando entrare            │
                                                          │
6. Backtest ──► Valida su dati storici                    │
                                                          │
7. Paper Trading ──► Demo account per 1-2 mesi            │
                                                          │
8. Live Trading ──► Con micro-lotti inizialmente          │
                                                          ▼
                                                       PROFIT 🎯
```

---

## ⚠️ Note Importanti

1. **MT5 richiede Windows** - Per download dati
2. **Non backtestare su dati di training** - Usa split temporale
3. **Inizia sempre su Demo** - Mai live senza test approfonditi
4. **Il mercato cambia** - Ritraining periodico necessario

---

## 📞 Supporto

Per domande o problemi, apri una nuova chat in questo progetto Claude.

---

*NexNow LTD - Forex AI Trading System v1.0*
