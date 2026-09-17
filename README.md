# Stock Direction Prediction

This project predicts whether a stock is likely to move upward or downward over the next 60 trading days using technical indicators and a machine learning model. It was built around an XGBoost classifier and a Streamlit web app for interactive prediction.

## Project Overview

The workflow includes:

- downloading market data from Yahoo Finance
- engineering technical indicators such as RSI, moving averages, volatility, Bollinger position, and return-based features
- training a direction prediction model
- exposing predictions through a Streamlit interface

This project is designed to make the stock prediction workflow easy to run and experiment with.

## Tech Stack

- Python 3.12
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib / Seaborn
- Streamlit
- yfinance

## Repository Structure

```bash
.
├── EGX30_Model_clean.ipynb      # Notebook containing exploratory analysis and model development
├── final_model.pkl              # Saved trained model artifact
├── stock_predictor.py           # Prediction logic and feature engineering
├── streamlit_app.py             # Streamlit web app
├── requirements.txt             # Python dependencies
├── pytest.ini                  # pytest configuration
├── tests/
│   └── test_stock_predictor.py # Basic validation tests
├── README.md                   # Project documentation
└── .venv                       # Local virtual environment
```

## Setup

Create and activate a virtual environment:

```bash
cd final_project
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the App

Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Model Behavior

The app predicts the stock direction for the next 60-day horizon using the same logic as the notebook:

- price returns
- moving average distances
- RSI
- MACD-based signals
- volatility ratios
- Bollinger band position
- ticker-based features

The app supports ticker input and a selected time window for historical data, then returns:

- Direction: Up / Down
- Confidence score
- Latest close price
- Last available date

## Example Usage

In the Streamlit sidebar:

- enter a ticker like `AAPL`
- choose a time period such as `6mo`, `1y`, or `2y`
- click Predict direction

## Testing

Run the validation tests with:

```bash
pytest -q
```

## Notes

- The saved model file is included for convenience, but the app can also fall back to a locally trained model when needed.
- This is a research-style stock prediction project and should be treated as an educational or exploratory tool rather than a guaranteed trading signal.

## License

This project is provided for educational and research use.

## Acknowledgements

- Yahoo Finance for market data access
- XGBoost for the classification model
- Streamlit for the interactive dashboard
