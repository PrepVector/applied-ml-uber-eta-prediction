import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class DataProcessing:
    def __init__(self):
        # The earth's radius (in km)
        self.R = 6371

    def update_column_name(self, df):
        """
        Update the column name 'Weatherconditions' to 'Weather_conditions' in the input DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        
        Returns:
        None
        """
        #print("Initial dataframe", df)
        # use the rename function of the pandas library and replace the column name to Weather_conditions in the same dataset
        #print("After", df.head())
        pass
        # Implementation details omitted


    def extract_feature_value(self, df):
        """
        Extract feature values from the input DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.

        
        Returns:
        None
        """
        #print("Initial dataframe", df)
        # use the rename function of the pandas library and replace the column name to Weather_conditions in the same dataset
        #print("After", df.head())
        pass
        # Implementation details omitted

    def extract_label_value(self, df):
        """
        Extract the label value 'Time_taken(min)' from the input DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        
        Returns:
        None
        """
        #print("Initial dataframe", df)
        # use the rename function of the pandas library and replace the column name to Weather_conditions in the same dataset
        #print("After", df.head())
        pass
        # Implementation details omitted

    def drop_columns(self, df):
        df.drop(['ID', 'Delivery_person_ID'], axis=1, inplace=True)

    def update_datatype(self, df):
        """
        Updates the data types of the following columns in the given DataFrame df:
        - 'Delivery_person_Age' to float64
        - 'Delivery_person_Ratings' to float64
        - 'multiple_deliveries' to float64
        - 'Order_Date' to datetime with format "%d-%m-%Y"
        """
        
        df['Delivery_person_Age'] = df['Delivery_person_Age'].astype('float64')
        df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype('float64')
        df['multiple_deliveries'] = df['multiple_deliveries'].astype('float64')
        df['Order_Date'] = pd.to_datetime(df['Order_Date'], format="%d-%m-%Y")

    def convert_nan(self, df):
        """
        Convert the string 'NaN' to a NaN value in the input DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.

        
        Returns:
        None        
        """
        #print("Initial dataframe", df)
        # use the rename function of the pandas library and replace the column name to Weather_conditions in the same dataset
        #print("After", df.head())
        pass
        # Implementation details omitted

    def handle_null_values(self, df):
        df['Delivery_person_Age'].fillna(np.random.choice(df['Delivery_person_Age']), inplace=True)
        df['Weather_conditions'].fillna(np.random.choice(df['Weather_conditions']), inplace=True)
        df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median(), inplace=True)
        df["Time_Orderd"] = df["Time_Orderd"].fillna(df["Time_Order_picked"])

        mode_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        mode_cols = ["Road_traffic_density",
                     "multiple_deliveries", "Festival", "City_type"]

        for col in mode_cols:
            df[col] = mode_imp.fit_transform(df[col].to_numpy().reshape(-1, 1)).ravel()

    def extract_date_features(self, df):
        """
        Extract date-related features from the input DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.

        
        Returns:
        None        
        """
        #print("Initial dataframe", df)
        # use the rename function of the pandas library and replace the column name to Weather_conditions in the same dataset
        #print("After", df.head())
        pass
        # Implementation details omitted

    def calculate_time_diff(self, df):
        """
        Calculates the time difference between order placement and order pickup in the given DataFrame df:
        - Converts 'Time_Orderd' and 'Time_Order_picked' to timedelta
        - Calculates 'Time_Order_picked_formatted' and 'Time_Ordered_formatted' based on 'Order_Date'
        - Calculates 'order_prepare_time' as the difference between 'Time_Order_picked_formatted' and 'Time_Ordered_formatted' in minutes
        - Fills null values in 'order_prepare_time' with the column median
        - Drops 'Time_Orderd', 'Time_Order_picked', 'Time_Ordered_formatted', 'Time_Order_picked_formatted', and 'Order_Date' columns
        """
        
        df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'])
        df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'])

        df['Time_Order_picked_formatted'] = df['Order_Date'] + pd.to_timedelta(np.where(df['Time_Order_picked'] < df['Time_Orderd'], 1, 0), unit='D') + df['Time_Order_picked']
        df['Time_Ordered_formatted'] = df['Order_Date'] + df['Time_Orderd']
        df['order_prepare_time'] = (df['Time_Order_picked_formatted'] - df['Time_Ordered_formatted']).dt.total_seconds() / 60

        df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace=True)
        df.drop(['Time_Orderd', 'Time_Order_picked', 'Time_Ordered_formatted', 'Time_Order_picked_formatted', 'Order_Date'], axis=1, inplace=True)

    def deg_to_rad(self, degrees):
        return degrees * (np.pi / 180)

    def distcalculate(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two latitude-longitude coordinates.

        Parameters:
        lat1 (float): Latitude of the first coordinate.
        lon1 (float): Longitude of the first coordinate.
        lat2 (float): Latitude of the second coordinate.
        lon2 (float): Longitude of the second coordinate.

        Returns:
        float: The distance between the two coordinates in kilometers.
        """
        #print("Initial dataframe", df)
        # use the rename function of the pandas library and replace the column name to Weather_conditions in the same dataset
        #print("After", df.head())
        pass
        # Implementation details omitted

    def calculate_distance(self, df):
        """
        Calculates the distance between the restaurant and delivery location in the given DataFrame df:
        - Creates a new 'distance' column
        - Calculates the distance using the 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', and 'Delivery_location_longitude' columns
        - Converts the 'distance' column to int64
        """
        
        df['distance'] = np.nan

        for i in range(len(df)):
            df.loc[i, 'distance'] = self.distcalculate(df.loc[i, 'Restaurant_latitude'],
                                                    df.loc[i, 'Restaurant_longitude'],
                                                    df.loc[i, 'Delivery_location_latitude'],
                                                    df.loc[i, 'Delivery_location_longitude'])
        df.distance = df.distance.astype("int64")

    def label_encoding(self, df):
        """
        Perform label encoding on categorical columns in the input DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.

        Returns:
        dict: A dictionary containing the label encoders for each categorical column.
        """
        #print("Initial dataframe", df)
        # use the rename function of the pandas library and replace the column name to Weather_conditions in the same dataset
        #print("After", df.head())
        pass
        # Implementation detail omitted

    def data_split(self, X, y):
        """
        Logic

        Returns:
        X_train, X_test, y_train, y_test        
        """
        #print("Initial dataframe", df)
        # use the rename function of the pandas library and replace the column name to Weather_conditions in the same dataset
        #print("After", df.head())
        pass
        # Implementation detail omitted

    def standardize(self, X_train, X_test):
        """
        Logic
        
        Returns:
        X_train, X_test, scaler
        """
        #print("Initial dataframe", df)
        # use the rename function of the pandas library and replace the column name to Weather_conditions in the same dataset
        #print("After", df.head())
        pass
        # Implementation detail omitted

    def cleaning_steps(self, df):
        self.update_column_name(df)
        self.extract_feature_value(df)
        self.drop_columns(df)
        self.update_datatype(df)
        self.convert_nan(df)
        self.handle_null_values(df)

    def perform_feature_engineering(self, df):
        self.extract_date_features(df)
        self.calculate_time_diff(df)
        self.calculate_distance(df)

    def evaluate_model(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("Mean Absolute Error (MAE):", round(mae, 2))
        print("Mean Squared Error (MSE):", round(mse, 2))
        print("Root Mean Squared Error (RMSE):", round(rmse, 2))
        print("R-squared (R2) Score:", round(r2, 2))
