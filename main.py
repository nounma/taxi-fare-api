from fastapi import FastAPI
from calculator import *

app = FastAPI()
app.state.model= load_model()
@app.get("/predict")


#X_new = pd.DataFrame(querystring)
#clean_querystring = querystring.clean_data

def clean_data(df):
    df = df.dropna(how='any', axis=0)
    try:
        df = df[df.fare_amount > 0]
        print(df.shape)
        df = df[df.fare_amount < 400]
    except:
      pass

    print(df.shape)
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0) | (df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    df = df[df.passenger_count > 0]
    df = df[df.passenger_count < 8]
    df = df[df["pickup_latitude"].between(left=40.5, right=40.9)]
    df = df[df["dropoff_latitude"].between(left=40.5, right=40.9)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-73.7)]
    df = df[df["dropoff_longitude"].between(left=-74.3, right=-73.7)]
    return df
    
    

def transform_time_features(X: pd.DataFrame) -> np.ndarray:
    assert isinstance(X, pd.DataFrame)
    
    timedelta = (X["pickup_datetime"] - pd.Timestamp('2009-01-01T00:00:00', tz='UTC'))/pd.Timedelta(1,'D')
    
    pickup_dt = X["pickup_datetime"].dt.tz_convert("America/New_York").dt
    dow = pickup_dt.weekday
    hour = pickup_dt.hour
    month = pickup_dt.month

    hour_sin = np.sin(2 * math.pi / 24 * hour)
    hour_cos = np.cos(2*math.pi / 24 * hour)
    
    return np.stack([hour_sin, hour_cos, dow, month, timedelta], axis=1)
def transform_lonlat_features(X: pd.DataFrame) -> pd.DataFrame:

    assert isinstance(X, pd.DataFrame)
    lonlat_features = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

    def distances_vectorized(df: pd.DataFrame, start_lat: str, start_lon: str, end_lat: str, end_lon: str) -> dict:
        """
        Calculate the haversine and Manhattan distances between two points on the earth (specified in decimal degrees).
        Vectorized version for pandas df
        Computes distance in Km
        """
        earth_radius = 6371

        lat_1_rad, lon_1_rad = np.radians(df[start_lat]), np.radians(df[start_lon])
        lat_2_rad, lon_2_rad = np.radians(df[end_lat]), np.radians(df[end_lon])

        dlon_rad = lon_2_rad - lon_1_rad
        dlat_rad = lat_2_rad - lat_1_rad

        manhattan_rad = np.abs(dlon_rad) + np.abs(dlat_rad)
        manhattan_km = manhattan_rad * earth_radius

        a = (np.sin(dlat_rad / 2.0)**2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon_rad / 2.0)**2)
        haversine_rad = 2 * np.arcsin(np.sqrt(a))
        haversine_km = haversine_rad * earth_radius

        return dict(
            haversine=haversine_km,
            manhattan=manhattan_km)

    result = pd.DataFrame(distances_vectorized(X, *lonlat_features))

    return result

def compute_geohash(X: pd.DataFrame, precision: int = 5) -> np.ndarray:
    """
    Add a GeoHash (ex: "dr5rx") of len "precision" = 5 by default
    corresponding to each (lon, lat) tuple, for pick-up, and drop-off
    """
    assert isinstance(X, pd.DataFrame)

    X["geohash_pickup"] = X.apply(
        lambda x: gh.encode(x.pickup_latitude, x.pickup_longitude, precision=precision),
        axis=1
    )
    X["geohash_dropoff"] = X.apply(
        lambda x: gh.encode(x.dropoff_latitude, x.dropoff_longitude, precision=precision),
        axis=1
    )

    return X[["geohash_pickup", "geohash_dropoff"]]



def main_prep_data(X_input):
    data_query_cache_path = "data_train_1k.csv"
    df = pd.read_csv(data_query_cache_path, parse_dates=['pickup_datetime'])
    df = clean_data(df)
    # Let's cap training set to reasonable values 
    df = df[df.fare_amount < 400]
    df = df[df.passenger_count < 8]
    X = df.drop("fare_amount", axis=1)
    y = df[["fare_amount"]]
    # PASSENGER PIPE

    def min_max_cust(p):
        p_min = 1
        p_max = 8
        return (p-p_min)/(p_max-p_min)
    passenger_pipe = FunctionTransformer(min_max_cust)

    # DISTANCE PIPE
    dist_min = 0
    dist_max = 100

    def min_max_dist(dist):
        dist_min = 0
        dist_max = 100
        return (dist - dist_min) / (dist_max - dist_min)

    distance_pipe = make_pipeline(
        FunctionTransformer(transform_lonlat_features),
        FunctionTransformer(min_max_dist)
    )

    # TIME PIPE
    timedelta_min = 0
    timedelta_max = 2090

    def min_max_time_delta(year):
        timedelta_min = 0
        timedelta_max = 2090
        return (year - timedelta_min) / (timedelta_max - timedelta_min)

    time_categories = [
        np.arange(0, 7, 1),  # days of the week
        np.arange(1, 13, 1)  # months of the year
    ]

    time_pipe = make_pipeline(
        FunctionTransformer(transform_time_features),
        make_column_transformer(
            (OneHotEncoder(
                categories=time_categories,
                sparse=False,
                handle_unknown="ignore"
            ), [2,3]), # corresponds to columns ["day of week", "month"], not the other columns

            (FunctionTransformer(min_max_time_delta), [4]), # min-max scale the columns 4 ["timedelta"]
            remainder="passthrough" # keep hour_sin and hour_cos
        )
    )

    # GEOHASH PIPE
    lonlat_features = [
        "pickup_latitude", "pickup_longitude", "dropoff_latitude",
        "dropoff_longitude"
    ]

    # Below are the 20 most frequent district geohashes of precision 5,
    # covering about 99% of all dropoff/pickup locations,
    # according to prior analysis in a separate notebook
    most_important_geohash_districts = [
        "dr5ru", "dr5rs", "dr5rv", "dr72h", "dr72j", "dr5re", "dr5rk",
        "dr5rz", "dr5ry", "dr5rt", "dr5rg", "dr5x1", "dr5x0", "dr72m",
        "dr5rm", "dr5rx", "dr5x2", "dr5rw", "dr5rh", "dr5x8"
    ]

    geohash_categories = [
        most_important_geohash_districts,  # pickup district list
        most_important_geohash_districts  # dropoff district list
    ]

    geohash_pipe = make_pipeline(
        FunctionTransformer(compute_geohash),
        OneHotEncoder(
            categories=geohash_categories,
            handle_unknown="ignore",
            sparse=False
        )
    )

    # COMBINED PREPROCESSOR
    final_preprocessor = ColumnTransformer(
        [
            ("passenger_scaler", passenger_pipe, ["passenger_count"]),
            ("time_preproc", time_pipe, ["pickup_datetime"]),
            ("dist_preproc", distance_pipe, lonlat_features),
            ("geohash", geohash_pipe, lonlat_features),
        ],
        n_jobs=-1,
    )


    final_preprocessor.fit(X)
    X_val=final_preprocessor.transform(X_input)
    return X_val

querystring_clean=clean_data(querystring)
querystring_clean=main_prep_data(querystring_clean)
querystring_clean

@app.get("/predict")
def predict(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,passenger_count):
    value_to_predict=[pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,passenger_count]
    model = app.state.model
    pred=model.predict([value_to_predict])[0]
    return {'prediction': int(pred)}