import os
import typing
from typing import Dict, List, Tuple, Any

# Set headless backend BEFORE importing pyplot to suppress GUI/threading warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import yfinance as yf

# Set visual style for plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

# ----------------------------------------------------------------------
# GLOBAL CONFIGURATION & PROJECT DIRECTORY
# ----------------------------------------------------------------------
TICKERS = ["NVDA", "SPY", "AAPL", "MSFT"]
INITIAL_CAPITAL = 10000.0  # $10,000 Starting Cash per asset

# Intraday % SL / TP Grid for 5m bars
SL_VALUES = [0.005, 0.010, 0.015, 0.020]  # 0.5%, 1.0%, 1.5%, 2.0%
TP_VALUES = [0.010, 0.015, 0.020, 0.030]  # 1.0%, 1.5%, 2.0%, 3.0%

# Overnight Mode Options: 
# 'partial' = 50% De-Risk at Close (Best for NVDA/MSFT)
# 'full'    = 100% Hold Overnight (Best for AAPL)
# 'none'    = 0% Hold / EOD Flatten (Best for SPY)
DEFAULT_OVERNIGHT_MODE = "partial"

PROJECT_DIR = "stock_backtest_project"
os.makedirs(PROJECT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. DATA FETCHING (60-DAY 5-MINUTE DATA)
# ----------------------------------------------------------------------
def fetch_5m_data(ticker: str) -> pd.DataFrame:
    """
    Downloads maximum 60 days of 5-minute interval OHLC data from yfinance.
    """
    print(f"\n[+] Fetching 5-minute data for {ticker} (Last 60 Days)...")
    try:
        df = yf.download(ticker, period="60d", interval="5m", auto_adjust=True, progress=False)
        if df.empty:
            print(f"[!] Warning: No 5-minute data downloaded for {ticker}.")
            return pd.DataFrame()
        
        # Flatten MultiIndex columns if returned by yfinance API
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                print(f"[!] Missing required column '{col}' for {ticker}.")
                return pd.DataFrame()

        df = df[required_cols].apply(pd.to_numeric, errors='coerce').dropna()
        return df

    except Exception as e:
        print(f"[!] Error downloading 5m data for {ticker}: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------------------
# 2. 12/26 SMA ZEROLINE CROSSOVER SIGNAL GENERATION (5-MINUTE BARS)
# ----------------------------------------------------------------------
def strategy_sma_zeroline_cross_5m(df: pd.DataFrame) -> pd.DataFrame:
    """
    12/26 SMA Zeroline Crossover Strategy on 5-minute bars:
    - Moving Line = 12-bar SMA - 26-bar SMA
    - Signal Line = 9-bar SMA of Moving Line
    - Long Entry: Moving Line crosses ABOVE Signal Line AND Moving Line < 0 AND Close > SMA 200
    - Short Entry: Moving Line crosses BELOW Signal Line AND Moving Line > 0 AND Close < SMA 200
    """
    df = df.copy()

    df['SMA_12'] = df['Close'].rolling(window=12).mean()
    df['SMA_26'] = df['Close'].rolling(window=26).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    df['Moving_Line'] = df['SMA_12'] - df['SMA_26']
    df['Signal_Line'] = df['Moving_Line'].rolling(window=9).mean()

    moving_above_signal = df['Moving_Line'] > df['Signal_Line']
    moving_above_signal_prev = df['Moving_Line'].shift(1) > df['Signal_Line'].shift(1)

    crossover_up = moving_above_signal & (~moving_above_signal_prev)
    crossover_down = (~moving_above_signal) & moving_above_signal_prev

    long_condition = (
        crossover_up 
        & (df['Moving_Line'] < 0) 
        & (df['Close'] > df['SMA_200'])
    )

    short_condition = (
        crossover_down 
        & (df['Moving_Line'] > 0) 
        & (df['Close'] < df['SMA_200'])
    )

    raw_signal = pd.Series(0, index=df.index)
    raw_signal[long_condition] = 1
    raw_signal[short_condition] = -1

    # Shift by 1 bar: Signal at close of bar t executes at Open of bar t+1
    df['Signal'] = raw_signal.shift(1).fillna(0).astype(int)
    return df

# ----------------------------------------------------------------------
# 3. BACKTEST ENGINE ($10,000 CAPITAL SIMULATION)
# ----------------------------------------------------------------------
def run_intraday_backtest(
    df: pd.DataFrame, 
    sl_pct: float, 
    tp_pct: float, 
    overnight_mode: str = "partial", 
    initial_cash: float = INITIAL_CAPITAL, 
    commission_per_share: float = 0.005
) -> Dict[str, Any]:
    cash = initial_cash
    position = 0  # +ve for Long, -ve for Short
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    is_partially_derisked = False

    portfolio_history = []
    trades = []

    dates = df.index
    n = len(df)

    for i in range(n):
        timestamp = dates[i]
        open_p = df['Open'].iloc[i]
        high_p = df['High'].iloc[i]
        low_p = df['Low'].iloc[i]
        close_p = df['Close'].iloc[i]
        current_signal = df['Signal'].iloc[i]

        # Identify EOD Market Close bar
        is_last_bar_of_day = False
        if i < n - 1:
            if dates[i+1].date() != timestamp.date():
                is_last_bar_of_day = True
        else:
            is_last_bar_of_day = True

        # Check if first bar of new trading day
        is_first_bar_of_day = False
        if i > 0 and dates[i].date() != dates[i-1].date():
            is_first_bar_of_day = True

        # --- A. RE-BUY 50% PORTION AT NEXT DAY OPEN (PARTIAL MODE) ---
        if overnight_mode == "partial" and is_first_bar_of_day and position != 0 and is_partially_derisked:
            if (position > 0 and current_signal != -1) or (position < 0 and current_signal != 1):
                target_allocation_cash = cash + abs(position * open_p)
                additional_shares = int((target_allocation_cash * 0.5) // (open_p + commission_per_share))
                if additional_shares > 0:
                    add_pos = additional_shares if position > 0 else -additional_shares
                    position += add_pos
                    cash -= add_pos * open_p + abs(add_pos) * commission_per_share
            is_partially_derisked = False

        # --- B. INTRA-BAR SL / TP & EOD OVERNIGHT HANDLING ---
        if position != 0:
            # 1. Long Exit
            if position > 0:
                hit_tp = high_p >= tp_price
                hit_sl = low_p <= sl_price
                if hit_tp or hit_sl:
                    exit_price = sl_price if hit_sl else tp_price
                    cash += (position * exit_price) - (position * commission_per_share)
                    trades.append({'pnl': (exit_price - entry_price) * position - (2 * position * commission_per_share)})
                    position = 0
                    is_partially_derisked = False

            # 2. Short Exit
            elif position < 0:
                hit_tp = low_p <= tp_price
                hit_sl = high_p >= sl_price
                if hit_tp or hit_sl:
                    exit_price = sl_price if hit_sl else tp_price
                    cash += (position * exit_price) - (abs(position) * commission_per_share)
                    trades.append({'pnl': (entry_price - exit_price) * abs(position) - (2 * abs(position) * commission_per_share)})
                    position = 0
                    is_partially_derisked = False

            # 3. EOD Overnight Rule
            if position != 0 and is_last_bar_of_day:
                if overnight_mode == "none":
                    exit_price = close_p
                    comm = abs(position) * commission_per_share
                    cash += (position * exit_price - comm)
                    pnl = (exit_price - entry_price) * position - comm if position > 0 else (entry_price - exit_price) * abs(position) - comm
                    trades.append({'pnl': pnl})
                    position = 0

                elif overnight_mode == "partial" and not is_partially_derisked:
                    shares_to_exit = int(position * 0.5)
                    if abs(shares_to_exit) > 0:
                        exit_price = close_p
                        comm = abs(shares_to_exit) * commission_per_share
                        cash += (shares_to_exit * exit_price - comm)
                        position -= shares_to_exit
                        is_partially_derisked = True

        # --- C. EXECUTE NEW SIGNALS AT BAR OPEN ---
        if current_signal != 0 and not (overnight_mode != "full" and is_last_bar_of_day):
            if (current_signal == 1 and position < 0) or (current_signal == -1 and position > 0):
                cash += (position * open_p) - (abs(position) * commission_per_share)
                position = 0
                is_partially_derisked = False

            # Enter Long Position
            if current_signal == 1 and position == 0:
                shares = int(cash // (open_p + commission_per_share))
                if shares > 0:
                    position = shares
                    entry_price = open_p
                    cash -= (shares * open_p + shares * commission_per_share)
                    tp_price = entry_price * (1.0 + tp_pct)
                    sl_price = entry_price * (1.0 - sl_pct)
                    is_partially_derisked = False

            # Enter Short Position
            elif current_signal == -1 and position == 0:
                shares = int(cash // (open_p + commission_per_share))
                if shares > 0:
                    position = -shares
                    entry_price = open_p
                    cash += (shares * open_p - shares * commission_per_share)
                    tp_price = entry_price * (1.0 - tp_pct)
                    sl_price = entry_price * (1.0 + sl_pct)
                    is_partially_derisked = False

        # Portfolio Valuation
        port_val = cash + (position * close_p)
        portfolio_history.append({'Timestamp': timestamp, 'Portfolio_Value': port_val})

    # Calculations
    port_df = pd.DataFrame(portfolio_history).set_index('Timestamp')
    port_df['Returns'] = port_df['Portfolio_Value'].pct_change().fillna(0.0)

    ending_balance = port_df['Portfolio_Value'].iloc[-1]
    total_return = (ending_balance - initial_cash) / initial_cash * 100.0
    std_ret = port_df['Returns'].std()
    
    # Annualized Sharpe for 5-minute bars
    sharpe_ratio = (port_df['Returns'].mean() / std_ret * np.sqrt(19656)) if std_ret > 0 else 0.0

    cum_max = port_df['Portfolio_Value'].cummax()
    drawdown = (port_df['Portfolio_Value'] - cum_max) / cum_max
    max_drawdown = drawdown.min() * 100.0 if not drawdown.empty else 0.0

    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        'Initial Capital': initial_cash,
        'Ending Balance': ending_balance,
        'Total Return (%)': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Win Rate (%)': win_rate,
        'Number of Trades': total_trades,
        'Portfolio_History': port_df
    }

# ----------------------------------------------------------------------
# 4. CHRONOLOGICAL TRAIN / TEST EXECUTION PIPELINE
# ----------------------------------------------------------------------
def process_intraday_ticker(ticker: str) -> Dict[str, Any]:
    raw_df = fetch_5m_data(ticker)
    if raw_df.empty:
        return {}

    df = strategy_sma_zeroline_cross_5m(raw_df).dropna()

    unique_dates = pd.Series(df.index.date).unique()
    if len(unique_dates) < 10:
        print(f"[!] Insufficient trading days for {ticker}.")
        return {}

    split_idx = len(unique_dates) // 2
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]

    # Use np.isin to prevent pandas ndarray AttributeError
    train_df = df[np.isin(df.index.date, train_dates)]
    test_df = df[np.isin(df.index.date, test_dates)]

    # 1. Train Optimization
    print(f"[+] Running Grid Search on First ~20 Days ({len(train_dates)} Days) for {ticker}...")
    grid_results = []
    for sl in SL_VALUES:
        for tp in TP_VALUES:
            res = run_intraday_backtest(train_df, sl_pct=sl, tp_pct=tp, overnight_mode=DEFAULT_OVERNIGHT_MODE)
            grid_results.append({'SL': sl, 'TP': tp, 'Sharpe Ratio': res['Sharpe Ratio']})

    grid_df = pd.DataFrame(grid_results)
    best_row = grid_df.sort_values(by='Sharpe Ratio', ascending=False).iloc[0]
    best_sl, best_tp = best_row['SL'], best_row['TP']

    print(f"    [★] Best SL: {best_sl*100:.1f}%, Best TP: {best_tp*100:.1f}% (Train Sharpe: {best_row['Sharpe Ratio']:.2f})")

    # 2. Out-of-Sample Evaluation
    print(f"[+] Running Out-of-Sample Test on Final ~20 Days ({len(test_dates)} Days) for {ticker}...")
    oos_res = run_intraday_backtest(test_df, sl_pct=best_sl, tp_pct=best_tp, overnight_mode=DEFAULT_OVERNIGHT_MODE)

    # 3. Save Plot
    save_intraday_plot(test_df, oos_res, ticker)

    return {
        'Ticker': ticker,
        'Initial Capital': f"${oos_res['Initial Capital']:,.2f}",
        'Ending Balance': f"${oos_res['Ending Balance']:,.2f}",
        'Total Return': f"{oos_res['Total Return (%)']:.2f}%",
        'Max Drawdown': f"{oos_res['Max Drawdown (%)']:.2f}%",
        'Sharpe Ratio': round(oos_res['Sharpe Ratio'], 2),
        'Win Rate': f"{oos_res['Win Rate (%)']:.1f}%",
        'Total Trades': oos_res['Number of Trades']
    }


def save_intraday_plot(test_df: pd.DataFrame, oos_res: Dict[str, Any], ticker: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    test_initial = test_df['Close'].iloc[0]
    buy_hold = (test_df['Close'] / test_initial) * INITIAL_CAPITAL

    ax.plot(oos_res['Portfolio_History'].index, oos_res['Portfolio_History']['Portfolio_Value'], label='5m SMA 12/26 Strategy', color='blue', lw=1.5)
    ax.plot(test_df.index, buy_hold, label=f'Buy & Hold ({ticker})', color='gray', linestyle='--', alpha=0.7)

    ax.set_title(f'{ticker}: 5m 12/26 SMA Strategy Out-of-Sample (Starting $10,000)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    plt.tight_layout()

    filename = os.path.join(PROJECT_DIR, f"{ticker}_5m_10k_intraday_results.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"    [->] Saved chart to: {filename}")

# ----------------------------------------------------------------------
# 5. MAIN EXECUTION PIPELINE
# ----------------------------------------------------------------------
def main():
    print("======================================================================")
    print("  5-MINUTE 12/26 SMA ZEROLINE BACKTEST ($10,000 STARTING CAPITAL)     ")
    print("======================================================================")

    summary_results = []
    for ticker in TICKERS:
        res = process_intraday_ticker(ticker)
        if res:
            summary_results.append(res)

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        print("\n\n====================================================================================================")
        print("                         60-DAY 5-MINUTE INTRADAY PERFORMANCE SUMMARY                              ")
        print("====================================================================================================")
        print(summary_df.to_string(index=False))
        print("====================================================================================================\n")

if __name__ == '__main__':
    main()