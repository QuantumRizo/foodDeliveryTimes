import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

TIME_FMT = '%H:%M'
DATE_FMT = '%d-%m-%Y'

def load_and_clean_data(path):
    df = pd.read_csv(path, parse_dates=['Order_Date'], infer_datetime_format=True, date_format=DATE_FMT, dayfirst=True)
    df.dropna(inplace=True)
    df.drop(['ID'], axis=1, inplace=True)
    df.rename(columns={'Time_taken (min)': 'Time_taken_min'}, inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df.sort_values(by=['order_date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def deduplicate(df):
    dup_c1 = ['delivery_person_id', 'delivery_person_age', 'order_date', 'time_orderd']
    df = df[~df.duplicated(subset=dup_c1, keep=False)]
    dup_c2 = ['delivery_person_id', 'delivery_person_age', 'delivery_person_ratings']
    df = df[~df.duplicated(subset=dup_c2, keep='first')]
    return df

def feature_engineering(df):
    df['delivery_person_age'] = df['delivery_person_age'].astype(int)
    df['time_orderd'] = pd.to_datetime(df['time_orderd'], format=TIME_FMT, errors='coerce')
    df['time_order_picked'] = pd.to_datetime(df['time_order_picked'], format=TIME_FMT, errors='coerce')
    df['pickup_time_min'] = (df['time_order_picked'] - df['time_orderd']).dt.total_seconds() / 60
    df = df.dropna(subset=['pickup_time_min'])
    df['pickup_time_min'] = df['pickup_time_min'].astype(np.int64)
    df.drop(['time_orderd', 'time_order_picked'], axis=1, inplace=True)
    return df

def get_distance(rlat, rlon, dlat, dlon):
    import geopy.distance
    res_coords = (abs(rlat), abs(rlon))
    dl_coords = (abs(dlat), abs(dlon))
    try:
        dist = geopy.distance.geodesic(res_coords, dl_coords).km
        return round(dist, 2)
    except:
        return None

def add_distance(df):
    df['delivery_dist_km'] = df.apply(lambda row: get_distance(
        row['restaurant_latitude'],
        row['restaurant_longitude'],
        row['delivery_location_latitude'],
        row['delivery_location_longitude']
    ), axis=1)
    unwanted_cols = ['restaurant_latitude', 'restaurant_longitude',
                     'delivery_location_latitude', 'delivery_location_longitude']
    df.drop(unwanted_cols, axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_date_features(df):
    df['month'] = df['order_date'].dt.month
    df['weekend'] = df['order_date'].dt.weekday.apply(lambda x: 'yes' if x >= 5 else 'no')
    df.drop('order_date', axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def encode_and_scale(df):
    df['multiple_deliveries'] = df['multiple_deliveries'].apply(lambda x: 'no' if x == 0.0 else 'yes')
    df.drop(['delivery_person_id', 'delivery_person_age', 'delivery_person_ratings'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    cat_cols = ['weather_conditions', 'road_traffic_density', 'type_of_order', 'type_of_vehicle',
                'multiple_deliveries', 'festival', 'city', 'weekend', 'vehicle_condition']
    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
    new_col_order = ['weather_conditions', 'road_traffic_density', 'vehicle_condition',
                     'type_of_order', 'type_of_vehicle', 'multiple_deliveries', 'festival', 'city',
                     'pickup_time_min', 'delivery_dist_km', 'month', 'weekend', 'time_taken_min']
    df = df[new_col_order]
    scaler = MinMaxScaler()
    num_cols = ['pickup_time_min', 'delivery_dist_km']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def preprocess(path):
    df = load_and_clean_data(path)
    df = deduplicate(df)
    df = feature_engineering(df)
    df = add_distance(df)
    df = add_date_features(df)
    df = encode_and_scale(df)
    return df