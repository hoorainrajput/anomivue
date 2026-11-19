import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
from anomaly_detector import train_model, detect_anomalies

def main():
    print("🚀 Starting Anomaly Detection System...")

    # Load data
    try:
        df = pd.read_csv("processed_trend_data.csv")
        print("✅ Data loaded!")
    except Exception as e:
        print("❌ Error:", e)
        return

    # Train model
    print("🧠 Training model...")
    train_model(df)

    # Detect anomalies
    print("🔍 Detecting anomalies...")
    result_df = detect_anomalies(df)

    # Show results
    print("\n📊 Sample Output:")
    print(result_df.head())

    # Save results
    result_df.to_csv("anomaly_output.csv", index=False)
    print("✅ Output saved!")

    # Count anomalies
    print(f"🚨 Total anomalies: {result_df['anomaly'].sum()}")

if __name__ == "__main__":
    import streamlit.runtime
    if streamlit.runtime.exists():
        import runpy
        import os
        dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
        runpy.run_path(dashboard_path, run_name="__main__")
    else:
        main()
