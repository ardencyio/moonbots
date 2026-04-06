
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest.strategies.open_gate import OpenGateStrategy

def generate_test_data():
    # Generate 60 minutes of data
    start_time = datetime(2024, 1, 1, 9, 30)
    data = []
    
    # 1. Gate Formation (9:30 - 9:35) - Range [100, 105]
    for i in range(5):
        data.append({
            "Open": 100 + i,
            "High": 105,
            "Low": 100,
            "Close": 101 + i,
            "Volume": 1000
        })
        
    # 2. Breakout (9:35 - 9:40) - Move to 110
    for i in range(5):
        data.append({
            "Open": 105 + i,
            "High": 106 + i,
            "Low": 104 + i,
            "Close": 106 + i,
            "Volume": 1000
        })
        
    # 3. Retest (9:40) - Touch 105 (gate_high)
    data.append({
        "Open": 110,
        "High": 110,
        "Low": 105, # Retest gate_high
        "Close": 107,
        "Volume": 1000
    })
    
    times = [start_time + timedelta(minutes=i) for i in range(len(data))]
    df = pd.DataFrame(data, index=times)
    return df

def test_long_tp_calculation():
    df = generate_test_data()
    config = {
        "gate_candle_minutes": 5,
        "stop_buffer_ticks": 1.0,
        "min_risk_reward": 2.0,
        "use_market_hours": False
    }
    strategy = OpenGateStrategy(config)
    
    print("Testing Long TP calculation...")
    
    # Run through the data
    for i in range(len(df)):
        row = df.iloc[i]
        signal = strategy.on_data(row, all_data=df.iloc[:i+1])
        
        if signal and signal["signal_type"] == "entry":
            print(f"Entry Signal at {df.index[i]}:")
            print(f"  Direction: {signal['direction']}")
            print(f"  Entry Price: {signal['entry_price']}")
            print(f"  Stop Loss: {signal['stop_loss']}")
            print(f"  Take Profit: {signal['take_profit']}")
            
            # Expected values:
            # Gate High: 105, Gate Low: 100
            # Entry: 107 (Close of retest bar)
            # Stop Loss: Gate Low (100) - Buffer (1.0) = 99.0
            # Risk: 107 - 99 = 8.0
            # TP: 107 + (8.0 * 2.0) = 107 + 16 = 123.0
            
            expected_tp = 123.0
            if abs(signal['take_profit'] - expected_tp) < 0.001:
                print("SUCCESS: TP calculation is correct for Long.")
            else:
                print(f"FAILURE: Expected TP {expected_tp}, got {signal['take_profit']}")
            return

    print("FAILURE: No entry signal generated")

def test_short_tp_calculation():
    # Generate data for short breakout
    start_time = datetime(2024, 1, 1, 9, 30)
    data = []
    
    # 1. Gate Formation (9:30 - 9:35) - Range [100, 105]
    for i in range(5):
        data.append({
            "Open": 105 - i,
            "High": 105,
            "Low": 100,
            "Close": 104 - i,
            "Volume": 1000
        })
        
    # 2. Breakout Down (9:35 - 9:40) - Move to 95
    for i in range(5):
        data.append({
            "Open": 100 - i,
            "High": 101 - i,
            "Low": 99 - i,
            "Close": 99 - i,
            "Volume": 1000
        })
        
    # 3. Retest (9:40) - Touch 100 (gate_low)
    data.append({
        "Open": 95,
        "High": 100, # Retest gate_low
        "Low": 95,
        "Close": 98,
        "Volume": 1000
    })
    
    times = [start_time + timedelta(minutes=i) for i in range(len(data))]
    df = pd.DataFrame(data, index=times)
    
    config = {
        "gate_candle_minutes": 5,
        "stop_buffer_ticks": 1.0,
        "min_risk_reward": 2.0,
        "use_market_hours": False
    }
    strategy = OpenGateStrategy(config)
    
    print("\nTesting Short TP calculation...")
    
    # Run through the data
    for i in range(len(df)):
        row = df.iloc[i]
        signal = strategy.on_data(row, all_data=df.iloc[:i+1])
        
        if signal and signal["signal_type"] == "entry":
            print(f"Entry Signal at {df.index[i]}:")
            print(f"  Direction: {signal['direction']}")
            print(f"  Entry Price: {signal['entry_price']}")
            print(f"  Stop Loss: {signal['stop_loss']}")
            print(f"  Take Profit: {signal['take_profit']}")
            
            # Expected values:
            # Gate High: 105, Gate Low: 100
            # Entry: 98 (Close of retest bar)
            # Stop Loss: Gate High (105) + Buffer (1.0) = 106.0
            # Risk: 106 - 98 = 8.0
            # TP: 98 - (8.0 * 2.0) = 98 - 16 = 82.0
            
            expected_tp = 82.0
            if abs(signal['take_profit'] - expected_tp) < 0.001:
                print("SUCCESS: TP calculation is correct for Short.")
            else:
                print(f"FAILURE: Expected TP {expected_tp}, got {signal['take_profit']}")
            return

    print("FAILURE: No entry signal generated")

if __name__ == "__main__":
    test_long_tp_calculation()
    test_short_tp_calculation()
