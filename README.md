# ForexML - Machine Learning Trading System

ForexML is an advanced, automated Forex trading system that leverages Machine Learning (ML) and Smart Money Concepts (SMC) to identify high-probability trading opportunities. It integrates directly with MetaTrader 5 (MT5) for data ingestion and order execution.

## Features

- **Advanced Data Ingestion:** Real-time tick streaming and OHLCV bar aggregation from MetaTrader 5.
- **SMC Structure Detection:** Automated identification of Market Structure Shifts (MSS), Order Blocks (OB), Fair Value Gaps (FVG), and Liquidity Pools.
- **ML-Powered Predictions:** Ensemble approach using LightGBM and PyTorch TabNet to predict market direction with confidence scoring.
- **Risk Management:** Sophisticated position sizing using Kelly fraction, daily drawdown limits, and correlated exposure checks.
- **Automated Lifecycle:** Handles the entire trade lifecycle from signal generation to execution and performance analytics.
- **Model Monitoring:** Built-in drift detection and automated retraining pipeline to maintain model performance over time.

## Architecture

The system is designed with a modular architecture:

- **System Bootstrap:** Initializes connections and configurations.
- **Market Data Ingestion:** Manages live data feeds.
- **SMC Detection:** Technical analysis layer based on Smart Money Concepts.
- **Feature Engineering:** Transforms raw data into ML-ready features.
- **Inference Engine:** Runs the ML models (LightGBM/TabNet).
- **Signal Generation:** Combines ML predictions with technical confluence.
- **Risk Management:** Validates trades against risk profiles.
- **Order Execution:** Communicates with MT5 to place orders.
- **Analytics & Monitoring:** Tracks performance and detects model drift.

Architectural diagrams are available in the `docs/` folder:
- `ForexML_ER.mmd`: Entity-Relationship diagram.
- `ForexML_FLOWCHART.mmd`: System process flowchart.

## Installation

### Prerequisites

- Python 3.8+
- MetaTrader 5 Terminal (Windows)
- An active MT5 Trading Account

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ForexML.git
   cd ForexML
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Usage

1. **Configure Environment Variables:**
   Set the following environment variables (or use a `.env` file):
   - `MT5_PATH`: Path to your `terminal64.exe`
   - `MT5_ACCOUNT`: Your MT5 account number
   - `MT5_PASSWORD`: Your MT5 password
   - `MT5_SERVER`: Your broker's server address

2. **Run the system:**
   ```bash
   python src/main.py
   ```

## Testing

To run the system in mock mode (without MT5), simply run the application. It will automatically fallback to mock data if MT5 is not found or cannot be initialized.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
