import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_model
from data_preprocessing import preprocess

def train(path):
    df = preprocess(path)
    categorical_cols = ['weather_conditions', 'road_traffic_density', 'vehicle_condition',
                        'type_of_order', 'type_of_vehicle', 'multiple_deliveries', 'festival', 'city', 'month', 'weekend']
    num_cols = ['pickup_time_min', 'delivery_dist_km']
    X = df[categorical_cols + num_cols]
    y = df['time_taken_min'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_cat = X_train[categorical_cols].values
    X_train_num = X_train[num_cols].values
    X_test_cat = X_test[categorical_cols].values
    X_test_num = X_test[num_cols].values
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)
    model = build_model(X_train_cat.shape[1], X_train_num.shape[1])
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_mae', mode='min', save_best_only=True, verbose=1)
    history = model.fit([X_train_cat, X_train_num], y_train,
                        validation_split=0.2, epochs=50, batch_size=32,
                        callbacks=[checkpoint], verbose=1)
    model.load_weights('best_model.h5')
    test_loss, test_mae = model.evaluate([X_test_cat, X_test_num], y_test)
    print(f"Best Test MAE: {test_mae:.2f}")
    return model, X_test_cat, X_test_num, y_test

if __name__ == "__main__":
    PATH = "../data/Zomato Dataset.csv"
    train(PATH)