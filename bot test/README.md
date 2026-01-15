# 🚀 Pro ML Trading Bot (Full-Stack)

A production-ready algorithmic trading system featuring **XGBoost ML predictions**, **Black-Scholes Options Pricing**, **0DTE Strategies**, and a **Django + Next.js** dashboard.

## 🌟 Key Features
- **Machine Learning Core**: XGBoost model trained on technical indicators and price action patterns.
- **Realistic Backtesting**: Built-in Black-Scholes engine simulates Options PnL with Delta & Theta decay.
- **0DTE Strategy**: Specialized logic for same-day expiration options (SPY/QQQ) capturing volatility jumps.
- **Risk Management**: Configurable Position Sizing, Stop Loss, and Take Profit rules.
- **Modern Dashboard**:
  - **Backend**: Django REST Framework API.
  - **Frontend**: Next.js 14 with Tailwind CSS & Shadcn UI (Dark/Light mode).
  - **Interactive**: Real-time equity curves, trade logs, and "One-Click" model retraining.

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Setup Backend
```bash
cd backend
pip install -r ../requirements.txt
python manage.py migrate
python manage.py runserver
```
*API will run at `http://localhost:8000`*

### 2. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```
*Dashboard will run at `http://localhost:3000`*

## 📈 Usage

### 🖥️ Dashboard (Recommended)
Navigate to **`http://localhost:3000/risk`** to:
1.  **Configure**: Set Balance ($1000), Risk (20%), and Toggle 0DTE.
2.  **Train**: Click "Retrain Models" to update ML models for all timeframes.
3.  **Simulate**: Click "Run Simulation" to see 1-Year backtest results instantly.

### 💻 CLI Mode
You can also run the bot from the terminal:

**Train Model:**
```bash
python main.py --symbol SPY --timeframe 15m --mode train
```

**Backtest (Standard):**
```bash
python main.py --symbol SPY --timeframe 15m --mode backtest --options
```

**Backtest (0DTE Strategy):**
```bash
python main.py --symbol SPY --timeframe 1m --zero-dte
```

## 🏗️ Project Structure
```
├── backend/            # Django API
│   ├── api/            # Endpoints (Train, Backtest)
│   └── core/           # Migrated Python Trading Logic
├── frontend/           # Next.js Dashboard
│   ├── src/app/        # Pages (Dashboard, Risk)
│   └── components/     # UI Components (Charts, Cards)
├── strategies/         # Strategy Logic
│   └── zero_dte.py     # 0DTE specific implementation
└── main.py             # CLI Orchestrator
```

## ⚠️ Risk Warning
Trading options, especially 0DTE, involves significant risk. This software is for educational and research purposes only.
