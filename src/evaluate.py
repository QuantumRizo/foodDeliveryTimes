import pandas as pd
import matplotlib.pyplot as plt
from train import train

def evaluate(path):
    model, X_test_cat, X_test_num, y_test = train(path)
    test_df = pd.DataFrame(X_test_num, columns=['pickup_time_min', 'delivery_dist_km'])
    # Add categorical columns (if needed, adjust accordingly)
    # test_df[categorical_cols] = X_test_cat
    test_df['time_taken_min'] = y_test
    test_df['predictions'] = model.predict([X_test_cat, X_test_num], verbose=0).flatten()
    plt.figure(figsize=(8, 6))
    plt.scatter(test_df['time_taken_min'], test_df['predictions'], alpha=0.6, color='dodgerblue', edgecolor='k')
    plt.plot([0, max(test_df['time_taken_min'])], [0, max(test_df['time_taken_min'])], color='red', linestyle='--')
    plt.xlabel("Actual Delivery Time (min)")
    plt.ylabel("Predicted Delivery Time (min)")
    plt.title("Actual vs Predicted Delivery Time")
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    PATH = "../data/Zomato Dataset.csv"
    evaluate(PATH)