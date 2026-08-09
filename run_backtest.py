import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

# ----------------------------------------------------------------------
# 1. DATA FETCHING & PREPARATION
# ----------------------------------------------------------------------
def fetch_and_prepare_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads adjusted OHLC data from yfinance and calculates technical indicators.
    Uses vectorization with shift(1) to eliminate look-ahead bias.
    """
    print(f"\n[+] Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        # auto_adjust=True returns split- and dividend-adjusted Open, High, Low, Close
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data retrieved for {ticker}.")
        
        # Flatten MultiIndex columns if present (yfinance modern API behavior)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize column names
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing expected column '{col}' in downloaded data.")

        # Ensure numeric type and remove NaNs
        df = df[required_cols].apply(pd.to_numeric, errors='coerce').dropna()

        # --------------------------------------------------------------
        # 2. INDICATOR CALCULATION (Vectorized)
        # --------------------------------------------------------------
        # Indicators calculated on Close
        df['SMA_9'] = df['Close'].rolling(window=9).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # Crossover logic: Compare today's SMA relation to yesterday's shift(1)
        sma9_above_20 = df['SMA_9'] > df['SMA_20']
        sma9_above_20_prev = df['SMA_9'].shift(1) > df['SMA_20'].shift(1)

        # 1: Bullish Crossover, -1: Bearish Crossover, 0: Neutral
        df['Crossover'] = 0
        df.loc[sma9_above_20 & (~sma9_above_20_prev), 'Crossover'] = 1
        df.loc[(~sma9_above_20) & sma9_above_20_prev, 'Crossover'] = -1

        # --------------------------------------------------------------
        # 3. ENTRY RULES (SMA200 Filter)
        # --------------------------------------------------------------
        df['Signal'] = 0
        # Long Entry: Bullish crossover & Close > SMA_200
        df.loc[(df['Crossover'] == 1) & (df['Close'] > df['SMA_200']), 'Signal'] = 1
        # Short Entry: Bearish crossover & Close < SMA_200
        df.loc[(df['Crossover'] == -1) & (df['Close'] < df['SMA_200']), 'Signal'] = -1

        return df.dropna()

    except Exception as e:
        print(f"[!] Error fetching/processing data for {ticker}: {e}")
        raise

# ----------------------------------------------------------------------
# 4 & 5. BACKTEST ENGINE (Single Run with Intra-day TP/SL & Margin Check)
# ----------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame, 
    sl_pct: float, 
    tp_pct: float, 
    initial_cash: float = 10000.0, 
    commission_per_share: float = 0.01
) -> Dict[str, Any]:
    """
    Simulates trading on a given DataFrame using specified SL and TP thresholds.
    Includes next-day open execution, intra-day High/Low TP/SL checks, and 25% margin checks.
    """
    cash = initial_cash
    position = 0  # Shares count: +ve for Long, -ve for Short
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0

    portfolio_history = []
    trades = []
    signal_history = []  

    dates = df.index
    n = len(df)

    for i in range(n):
        date = dates[i]
        open_p = df['Open'].iloc[i]
        high_p = df['High'].iloc[i]
        low_p = df['Low'].iloc[i]
        close_p = df['Close'].iloc[i]
        
        # Today's signal (generated at close of previous day)
        current_signal = df['Signal'].iloc[i]
        
        # --- A. CHECK INTRA-DAY SL / TP & MARGIN SAFETY FOR EXISTING POSITIONS ---
        if position != 0:
            notional_value = abs(position * close_p)
            current_portfolio_val = cash + (position * close_p)
            
            # Margin Check (25% Maintenance Margin)
            if current_portfolio_val < 0.25 * notional_value:
                # Forced Market Exit at Open of today
                exit_price = open_p
                revenue = position * exit_price
                comm = abs(position) * commission_per_share
                cash += (revenue - comm)
                
                trades.append({
                    'type': 'LONG' if position > 0 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'reason': 'MARGIN_CALL',
                    'pnl': (exit_price - entry_price) * position - comm
                })
                position = 0

            # Long Position Exit Check
            elif position > 0:
                hit_tp = high_p >= tp_price
                hit_sl = low_p <= sl_price
                
                if hit_tp or hit_sl:
                    # If both hit on same bar, assume pessimistic path (Stop-Loss first)
                    exit_price = sl_price if hit_sl else tp_price
                    reason = 'SL' if hit_sl else 'TP'
                    
                    cash += (position * exit_price) - (position * commission_per_share)
                    trades.append({
                        'type': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'reason': reason,
                        'pnl': (exit_price - entry_price) * position - (2 * position * commission_per_share)
                    })
                    position = 0

            # Short Position Exit Check
            elif position < 0:
                hit_tp = low_p <= tp_price
                hit_sl = high_p >= sl_price
                
                if hit_tp or hit_sl:
                    exit_price = sl_price if hit_sl else tp_price
                    reason = 'SL' if hit_sl else 'TP'
                    
                    cash += (position * exit_price) - (abs(position) * commission_per_share)
                    trades.append({
                        'type': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'reason': reason,
                        'pnl': (entry_price - exit_price) * abs(position) - (2 * abs(position) * commission_per_share)
                    })
                    position = 0

        # --- B. EXECUTE NEW ENTRY / REVERSE SIGNALS AT TODAY'S OPEN ---
        if current_signal != 0:
            # Reverse existing position if opposite signal appears and not already exited
            if (current_signal == 1 and position < 0) or (current_signal == -1 and position > 0):
                cash += (position * open_p) - (abs(position) * commission_per_share)
                pnl = (entry_price - open_p) * abs(position) if position < 0 else (open_p - entry_price) * position
                trades.append({
                    'type': 'SHORT' if position < 0 else 'LONG',
                    'entry_price': entry_price,
                    'exit_price': open_p,
                    'reason': 'SIGNAL_REVERSAL',
                    'pnl': pnl - (2 * abs(position) * commission_per_share)
                })
                position = 0

            # Enter new Long Position
            if current_signal == 1 and position == 0:
                shares = int(cash // (open_p + commission_per_share))
                if shares > 0:
                    position = shares
                    entry_price = open_p
                    cash -= (shares * open_p + shares * commission_per_share)
                    tp_price = entry_price * (1.0 + tp_pct)
                    sl_price = entry_price * (1.0 - sl_pct)
                    signal_history.append((date, 'BUY', open_p))

            # Enter new Short Position
            elif current_signal == -1 and position == 0:
                shares = int(cash // (open_p + commission_per_share))
                if shares > 0:
                    position = -shares
                    entry_price = open_p
                    cash += (shares * open_p - shares * commission_per_share)
                    tp_price = entry_price * (1.0 - tp_pct)
                    sl_price = entry_price * (1.0 + sl_pct)
                    signal_history.append((date, 'SELL', open_p))

        # --- C. DAILY PORTFOLIO VALUATION ---
        port_val = cash + (position * close_p)
        portfolio_history.append({'Date': date, 'Portfolio_Value': port_val})

    # --- D. PERFORMANCE METRICS CALCULATION ---
    port_df = pd.DataFrame(portfolio_history).set_index('Date')
    port_df['Returns'] = port_df['Portfolio_Value'].pct_change().fillna(0.0)

    total_return = (port_df['Portfolio_Value'].iloc[-1] - initial_cash) / initial_cash * 100.0

    # Annualized Sharpe Ratio (assuming 252 trading days, 0% Risk-Free Rate)
    std_ret = port_df['Returns'].std()
    sharpe_ratio = (port_df['Returns'].mean() / std_ret * np.sqrt(252)) if std_ret != 0 else 0.0

    # Max Drawdown
    cum_max = port_df['Portfolio_Value'].cummax()
    drawdown = (port_df['Portfolio_Value'] - cum_max) / cum_max
    max_drawdown = drawdown.min() * 100.0

    # Win Rate & Trade Counts
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        'SL': sl_pct,
        'TP': tp_pct,
        'Total Return (%)': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Win Rate (%)': win_rate,
        'Number of Trades': total_trades,
        'Portfolio_History': port_df,
        'Trades': trades,
        'Signals': signal_history
    }

# ----------------------------------------------------------------------
# 6 & 7. GRID SEARCH & TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
def perform_grid_search(df_train: pd.DataFrame, sl_values: List[float], tp_values: List[float]) -> pd.DataFrame:
    """Runs a grid search over all combinations of SL and TP parameters on training data."""
    results = []
    print(f"\n[+] Starting Grid Search on Training Set ({len(sl_values) * len(tp_values)} combinations)...")
    
    for sl in sl_values:
        for tp in tp_values:
            res = run_backtest(df_train, sl_pct=sl, tp_pct=tp)
            results.append({
                'SL': sl,
                'TP': tp,
                'Total Return (%)': res['Total Return (%)'],
                'Sharpe Ratio': res['Sharpe Ratio'],
                'Max Drawdown (%)': res['Max Drawdown (%)'],
                'Win Rate (%)': res['Win Rate (%)'],
                'Number of Trades': res['Number of Trades']
            })

    return pd.DataFrame(results)

# ----------------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ----------------------------------------------------------------------
def main():
    ticker = 'NVDA'  # Change to 'SPY' or any desired ticker
    full_df = fetch_and_prepare_data(ticker, start_date='2010-01-01', end_date='2026-08-09')

    # Train / Test Split
    train_df = full_df.loc['2010-01-01':'2020-12-31']
    test_df = full_df.loc['2021-01-01':'2026-08-09']

    print(f"Data Split Success: Train size = {len(train_df)} bars, Test size = {len(test_df)} bars.")

    # Grid Search Parameters
    sl_values = [0.03, 0.05, 0.07, 0.10]
    tp_values = [0.06, 0.08, 0.10, 0.14, 0.20]

    # Run Optimization
    grid_results = perform_grid_search(train_df, sl_values, tp_values)
    
    # Display Grid Search Table sorted by Sharpe Ratio
    grid_sorted = grid_results.sort_values(by='Sharpe Ratio', ascending=False)
    print("\n=== TOP 5 GRID SEARCH RESULTS (TRAINING DATA) ===")
    print(grid_sorted.head().to_string(index=False))

    # Find Best Combination
    best_row = grid_sorted.iloc[0]
    best_sl = best_row['SL']
    best_tp = best_row['TP']

    print(f"\n[★] OPTIMAL PARAMETERS FOUND:")
    print(f"    Optimal SL: {best_sl*100:.1f}% | Optimal TP: {best_tp*100:.1f}%")
    print(f"    In-Sample Sharpe Ratio: {best_row['Sharpe Ratio']:.2f}")

    # Out-of-Sample Evaluation
    print("\n[+] Running Out-of-Sample Backtest (2021-2026)...")
    oos_result = run_backtest(test_df, sl_pct=best_sl, tp_pct=best_tp)

    print("\n=== OUT-OF-SAMPLE PERFORMANCE METRICS ===")
    print(f"Optimal SL found: {best_sl*100:.1f}%, Optimal TP found: {best_tp*100:.1f}%. Out-of-sample Sharpe: {oos_result['Sharpe Ratio']:.2f}")
    print(f"Total Return: {oos_result['Total Return (%)']:.2f}%")
    print(f"Max Drawdown: {oos_result['Max Drawdown (%)']:.2f}%")
    print(f"Win Rate: {oos_result['Win Rate (%)']:.2f}%")
    print(f"Total Trades Executed: {oos_result['Number of Trades']}")

    # ------------------------------------------------------------------
    # 8. VISUALIZATION & OUTPUT
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)

    # 1. Heatmap of Sharpe Ratios (In-Sample Grid Search)
    ax_heat = fig.add_subplot(gs[0, 0])
    heatmap_data = grid_results.pivot(index='SL', columns='TP', values='Sharpe Ratio')
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu", 
        ax=ax_heat,
        cbar_kws={'label': 'Sharpe Ratio'}
    )
    ax_heat.set_title(f'Training Set Sharpe Ratio Heatmap ({ticker})', fontsize=12, fontweight='bold')
    ax_heat.set_xlabel('Take-Profit (TP)')
    ax_heat.set_ylabel('Stop-Loss (SL)')

    # Calculate Buy & Hold Strategy for Out-of-Sample Benchmark
    test_initial_price = test_df['Close'].iloc[0]
    buy_and_hold = (test_df['Close'] / test_initial_price) * 10000.0

    # 2. Equity Curve Comparison (Out-of-Sample)
    ax_equity = fig.add_subplot(gs[0, 1])
    ax_equity.plot(oos_result['Portfolio_History'].index, oos_result['Portfolio_History']['Portfolio_Value'], label='Optimized Strategy', color='blue', lw=2)
    ax_equity.plot(test_df.index, buy_and_hold, label=f'Buy & Hold ({ticker})', color='gray', linestyle='--', alpha=0.8)
    ax_equity.set_title('Out-of-Sample Equity Curve (2021 - 2026)', fontsize=12, fontweight='bold')
    ax_equity.set_ylabel('Portfolio Value ($)')
    ax_equity.legend()

    # 3. Price Chart with Trade Entry Markers
    ax_trades = fig.add_subplot(gs[1, :])
    ax_trades.plot(test_df.index, test_df['Close'], label=f'{ticker} Close Price', color='black', alpha=0.5, lw=1)
    ax_trades.plot(test_df.index, test_df['SMA_200'], label='200 SMA', color='orange', linestyle=':', lw=1.5)

    # Annotate Trades
    signals = oos_result['Signals']
    buy_dates = [s[0] for s in signals if s[1] == 'BUY']
    buy_prices = [s[2] for s in signals if s[1] == 'BUY']
    sell_dates = [s[0] for s in signals if s[1] == 'SELL']
    sell_prices = [s[2] for s in signals if s[1] == 'SELL']

    if buy_dates:
        ax_trades.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Long Entry', zorder=5)
    if sell_dates:
        ax_trades.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Short Entry', zorder=5)

    ax_trades.set_title('Out-of-Sample Price & Trade Entry Signals', fontsize=12, fontweight='bold')
    ax_trades.set_ylabel('Price ($)')
    ax_trades.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()