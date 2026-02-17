import pandas as pd

def main():
    df = pd.read_csv('data/processed/processed_with_target.csv')
    counts = df['is_high_risk'].value_counts()
    print('is_high_risk value counts:')
    print(counts)
    print(f"\nTotal rows: {len(df)}")
    print(f"Proportion high risk: {counts.get(1,0)/len(df):.4f}")
    print(f"Proportion low risk: {counts.get(0,0)/len(df):.4f}")

if __name__ == "__main__":
    main()
