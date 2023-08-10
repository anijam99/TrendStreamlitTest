import streamlit as st
import joblib
import concurrent.futures
from snowflake.snowpark.session import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark.window import Window
from sklearn import preprocessing # https://github.com/Snowflake-Labs/snowpark-python-demos/tree/main/sp4py_utilities
from snowflake.snowpark.functions import col
import plotly.express as px
import getpass
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import math
import plotly.graph_objects as go

import folium
from streamlit_folium import st_folium
import openrouteservice as ors
import operator
from functools import reduce
from streamlit_javascript import st_javascript

st.set_page_config(layout="wide")

#Loading model and data 
ayrton_model=joblib.load('ayrton_model.joblib')
javier_model=joblib.load('javier_model.joblib')
minh_model=joblib.load('minh_model.joblib')
nathan_model=joblib.load('nathan_model.joblib')
vibu_model=joblib.load('vibu_model.joblib')
old_updated_model=joblib.load('updated_old_model.joblib')
old_model=joblib.load('model.joblib')
model=joblib.load('group_model.joblib')
connection_parameters = { "account": 'hiioykl-ix77996',"user": 'JAVIER',"password": '02B289223r04', "role": "ACCOUNTADMIN","database": "FROSTBYTE_TASTY_BYTES","warehouse": "COMPUTE_WH"}

session = Session.builder.configs(connection_parameters).create()

import plotly.express as px
st.title("SpeedyBytes 🚚 - Minh's Model")

list_of_tabs = ["Revenue Forecasting & Model Performance"]
tabs = st.tabs(list_of_tabs)

#Code to get the updated model from asg2
def updated_old_model():
    session.use_schema("ANALYTICS")
    X_final_scaled=session.sql('Select * from "Sales_Forecast_Training_Data"').to_pandas()
    X_final_scaled.rename(columns={"Profit": "Revenue"},inplace=True)

    X_final_scaled = trim_outliers(X_final_scaled, 'Revenue')

    outliers_IV = np.where(X_final_scaled['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'] >1.7, True, np.where(X_final_scaled['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'] < -1, True, False))
    X_final_scaled = X_final_scaled.loc[~outliers_IV]
    outliers_IV = np.where(X_final_scaled['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'] >0.7, True, np.where(X_final_scaled['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'] < -0.7, True, False))
    X_final_scaled = X_final_scaled.loc[~outliers_IV]

    # Split the dataset into features (X) and target (y)
    X = X_final_scaled.drop("Revenue",axis=1)
    y = X_final_scaled["Revenue"]
    # Split the dataset into training and testing datasets
    X_training, X_holdout, y_training, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.2, random_state=42)
    xgb = XGBRegressor(objective="reg:squarederror", learning_rate=0.01523, max_depth=9, colsample_bytree=0.578, n_estimators=641, subsample=0.854)
    xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)])
    print('Train MSE is: ', mean_squared_error(xgb.predict(X_train), y_train))
    print('Test MSE is: ', mean_squared_error(xgb.predict(X_test), y_test))
    print()
    print('Train RMSE is: ',  math.sqrt(mean_squared_error(xgb.predict(X_train), y_train)))
    print('Test RMSE is: ', math.sqrt(mean_squared_error(xgb.predict(X_test), y_test)))
    print()
    print('Train MAE is: ', mean_absolute_error(xgb.predict(X_train), y_train))
    print('Test MAE is: ', mean_absolute_error(xgb.predict(X_test), y_test))
    print()
    print('Train R2 is: ', r2_score(xgb.predict(X_train), y_train))
    print('Test R2 is: ', r2_score(xgb.predict(X_test), y_test))
    print('Holdout MSE is: ', mean_squared_error(df_predictions['Predicted'], df_predictions['Holdout']))
    print()
    print('Holdout RMSE is: ',  math.sqrt(mean_squared_error(df_predictions['Predicted'], df_predictions['Holdout'])))
    print()
    print('Holdout MAE is: ', mean_absolute_error(df_predictions['Predicted'], df_predictions['Holdout']))
    print()
    print('Holdout R2 is: ', r2_score(df_predictions['Predicted'], df_predictions['Holdout']))
    joblib.dump(xgb, 'updated_old_model.joblib')

#TRIMMING CODE
def trim_outliers(dataframe, column, lower_percentile=0.01, upper_percentile=0.99):
    lower_bound = dataframe[column].quantile(lower_percentile)
    upper_bound = dataframe[column].quantile(upper_percentile)
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

from sklearn.model_selection import GridSearchCV
def train_javier_model():
    xgb = XGBRegressor(objective= 'reg:squarederror',
    learning_rate= 0.0125, 
    max_depth= 7,
    colsample_bytree= 0.65, 
    n_estimators= 751,  
    subsample= 0.9,  
    min_child_weight= 5, 
    gamma= 0.2,  
    )
    xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)])
    joblib.dump(xgb, 'javier_model.joblib')

def train_minh_model():
    def log_transform(dataframe, column):
        dataframe[column] = np.log1p(dataframe[column])
        return dataframe
    X_final_scaled = log_transform(X_final_scaled, 'Revenue')
    # Split the dataset into features (X) and target (y)
    X = X_final_scaled.drop("Revenue",axis=1)
    y = X_final_scaled["Revenue"]
    # Split the dataset into training and testing datasets
    X_training, X_holdout, y_training, y_holdout = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.2)

    # Define the grid of hyperparameters to search
    param_grid = {
        'learning_rate': [0.01, 0.015, 0.02],
        'max_depth': [6, 7, 8, 9],
        'colsample_bytree': [0.5, 0.6, 0.7],
        'n_estimators': [600, 700, 800],
        'subsample': [0.8, 0.9, 0.95],
        'min_child_weight': [2, 3, 4],
        'gamma': [0.1, 0.2]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and corresponding score
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Score:", -grid_search.best_score_)  # Negative of mean squared error

    xgb = XGBRegressor(objective= 'reg:squarederror',
    learning_rate= 0.015,
    max_depth= 8, 
    colsample_bytree= 0.6,
    n_estimators= 700,  
    subsample= 0.9,  
    min_child_weight= 3, 
    gamma= 0.1
    )
    xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)])
    joblib.dump(xgb, 'minh_model.joblib')

def train_ayrton_model():    
    xgb = XGBRegressor(objective= 'reg:squarederror',
    learning_rate= 0.01,
    max_depth= 10,
    colsample_bytree= 0.6,
    n_estimators= 1200,
    subsample= 0.9,
    min_child_weight= 5,
    gamma= 0.1)
    xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)])
    joblib.dump(xgb, 'ayrton_model.joblib')

def train_nathan_model():
    xgb = XGBRegressor(objective= 'reg:squarederror',
    learning_rate= 0.005,
    max_depth= 8,
    colsample_bytree= 0.8,
    n_estimators= 1000,
    subsample= 0.75,
    min_child_weight= 1,
    gamma= 0.2
    )
    xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)])
    joblib.dump(xgb, 'nathan_model.joblib')

def train_vibu_model():
    xgb = XGBRegressor(objective='reg:squarederror',
    learning_rate= 0.01,
    max_depth= 6,
    colsample_bytree= 0.7,
    n_estimators= 800,
    subsample= 0.85,
    min_child_weight= 3,
    gamma= 0.3
    )
    xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)])
    joblib.dump(xgb, 'vibu_model.joblib')


with tabs[0]: #Tran Huy Minh S10223485H Tab Revenue Forecasting & Model Performance
    import calendar
    def get_dates(year, month):
    
        """ Function Documentation
            Generate a list of dates for the given month and year.
    
            Returns:
                list: A list of date strings in the format 'YYYY-MM-DD'.
            """
    
        try:
            # Get the number of days in the given month
            num_days = calendar.monthrange(year, month)[1]
    
            # Generate a list of dates for the given month and year
            dates = [f"{year}-{month:02d}-{day:02d}" for day in range(1, num_days + 1)]
    
            return dates
    
        except Exception as e:
            print(f"An error occurred while retrieving list of dates: {e}")
            return pd.DataFrame()
            
    import datetime
    import holidays
    def add_public_holiday_column(df, date_column): #ONLY USED IN CONJUCTION WITH upload_input_data_to_snowflake() function
    
        """ Function Documentation
            Add a column to the DataFrame indicating whether each date is a public holiday using imported library.
    
            Returns:
                pandas.DataFrame: The DataFrame with an additional column "PUBLIC_HOLIDAY".
            """
    
        try:
            # Create an instance of the holiday class for the appropriate country
            country_code = 'US'  # Replace with the appropriate country code
            holiday_list = holidays.CountryHoliday(country_code)
    
            # Convert the date column to datetime if it's not already in that format
            df[date_column] = pd.to_datetime(df[date_column])
    
            # Create a new column "PUBLIC_HOLIDAY" and set initial values to 0
            df['PUBLIC_HOLIDAY'] = 0
    
            # Iterate over each date in the date column
            for date in df[date_column]:
                # Check if the date is a public holiday
                if date in holiday_list:
                    # Set the value of "PUBLIC_HOLIDAY" to 1 if it is a public holiday
                    df.loc[df[date_column] == date, 'PUBLIC_HOLIDAY'] = 1
    
            return df
    
        except Exception as e:
            print(f"An error occurred while retrieving public holiday information: {e}")
            return pd.DataFrame()
    
    def get_location_id(truck_id): #ONLY USED IN CONJUCTION WITH upload_input_data_to_snowflake() function
        """
        Get location ID, city, and region for a given truck ID.
    
        Returns:
            pandas.DataFrame: A DataFrame containing the location ID, city, and region for the given truck ID.
        """
        try:
            # Set the schema to "RAW_POS"
            session.use_schema("RAW_POS")
    
            # Query the Truck table to get the city for the given truck ID
            query = "SELECT PRIMARY_CITY FROM TRUCK WHERE TRUCK_ID = {}".format(truck_id)
            city_df = session.sql(query).toPandas()
            city = city_df['PRIMARY_CITY'].iloc[0]
    
            # Query the Location table to get the location ID for the city
            query = "SELECT LOCATION_ID FROM LOCATION WHERE CITY = '{}'".format(city)
            location_df = session.sql(query).toPandas()
    
            # Add the truck ID to the DataFrame
            location_df['TRUCK_ID'] = truck_id
    
            return location_df
    
        except Exception as e:
            print(f"An error occurred while retrieving location information: {e}")
            return pd.DataFrame()
    
    def get_hours_df(truck_id):
        """
        Get a DataFrame with hours and the corresponding truck ID.
    
        Returns:
            pandas.DataFrame: A DataFrame containing the truck ID and hours from 0 to 23.
        """
        try:
            # Create a list of hours from 0 to 23
            hours = list(range(24))
    
            # Create a dictionary with column names and corresponding data
            data = {'TRUCK_ID': [truck_id] * 24, 'HOUR': hours}
    
            # Create a new DataFrame from the dictionary
            new_df = pd.DataFrame(data)
    
            return new_df
    
        except Exception as e:
            print(f"An error occurred while generating the hours DataFrame: {e}")
            return pd.DataFrame()
    
    def upload_input_data_to_snowflake():
        """
        Uploads input data to Snowflake for the food truck revenue trend forecast.
    
        This function performs various data preprocessing and joins, and then writes the final DataFrame
        containing input data to the "Trend_Input_Data" table in the "ANALYTICS" schema of the "FROSTBYTE_TASTY_BYTES" database.
    
        Note: This function is intended for one-time use to upload data to Snowflake.
    
        Returns:
            None
        """
        try:
            # Set the schema to "ANALYTICS"
            session.use_schema("ANALYTICS")
    
            # Load data from the "Sales_Forecast_Training_Data" table
            X_final_scaled = session.sql('Select * from "Sales_Forecast_Training_Data"').to_pandas()
            X_final_scaled.rename(columns={"Profit": "Revenue"}, inplace=True)
    
            # Load data from the "ANALYTICS.SALES_PREDICTION" table
            sales_pred = session.sql("select * from ANALYTICS.SALES_PREDICTION").to_pandas()
    
            # Merge the dataframes based on the "l_w5i8_DATE" column
            X_final_scaled = X_final_scaled.merge(sales_pred["l_w5i8_DATE"].astype(str).str[:4].rename('YEAR'), left_index=True, right_index=True)
    
            # Filter data for specific truck IDs and years
            truck_ids = [27, 28, 43, 44, 46, 47]
            years = ['2020', '2021', '2022']
            X_final_scaled = X_final_scaled[(X_final_scaled['TRUCK_ID'].isin(truck_ids)) & (X_final_scaled['YEAR'].isin(years))]
    
            # Set the schema to "ANALYTICS"
            session.use_schema("ANALYTICS")
    
            # Load data from the "weadf_trend" table
            weadf = session.sql('select * from "weadf_trend"').to_pandas()
            weadf['DATE'] = pd.to_datetime(weadf['DATE'])
    
            # Process data for each truck, year, and month
            for truck in truck_ids:
                for year in years:
                    for month in range(1, 13):
                        current_df = X_final_scaled[
                            (X_final_scaled['TRUCK_ID'] == truck) & 
                            (X_final_scaled['MONTH'] == month) & 
                            (X_final_scaled['YEAR'] == year)
                        ]
    
                        # Generate dates for the given month and year
                        current_dates = get_dates(int(year), month)
                        main_df = pd.DataFrame({'TRUCK_ID': [truck] * len(current_dates), 'DATE': current_dates})
    
                        # Add public holiday column to main_df
                        main_df = add_public_holiday_column(main_df, 'DATE')
    
                        # Join location data
                        main_df = pd.merge(main_df, get_location_id(truck), how='left', left_on='TRUCK_ID', right_on='TRUCK_ID').drop_duplicates()
    
                        # Join hours data
                        main_df = pd.merge(main_df, get_hours_df(truck), how='left', left_on='TRUCK_ID', right_on='TRUCK_ID').drop_duplicates()
    
                        # Join weather data
                        main_df = pd.merge(main_df, weadf,  how='left', left_on=['LOCATION_ID', 'HOUR', 'DATE'], right_on=['LOCATION_ID', 'H', 'DATE']).drop_duplicates()
                        main_df = main_df.drop('H', axis=1).drop_duplicates().dropna()
    
                        # Additional data preprocessing
                        main_df['DATE'] = pd.to_datetime(main_df['DATE'])
                        main_df['MONTH'] = main_df['DATE'].dt.month
                        main_df['DOW'] = main_df['DATE'].dt.weekday
                        main_df['DAY'] = main_df['DATE'].dt.day
                        main_df['YEAR'] = main_df['DATE'].dt.year
                        main_df['DATE'] = main_df['DATE'].astype(str)
    
                        # Join encoded data
                        encoded_X = current_df[['TRUCK_ID', 'MENU_TYPE_GYROS_ENCODED', 'MENU_TYPE_CREPES_ENCODED', 'MENU_TYPE_BBQ_ENCODED', 'MENU_TYPE_SANDWICHES_ENCODED', 'MENU_TYPE_Mac & Cheese_encoded', 'MENU_TYPE_POUTINE_ENCODED', 'MENU_TYPE_ETHIOPIAN_ENCODED', 'MENU_TYPE_TACOS_ENCODED', 'MENU_TYPE_Ice Cream_encoded', 'MENU_TYPE_Hot Dogs_encoded', 'MENU_TYPE_CHINESE_ENCODED', 'MENU_TYPE_Grilled Cheese_encoded', 'MENU_TYPE_VEGETARIAN_ENCODED', 'MENU_TYPE_INDIAN_ENCODED', 'MENU_TYPE_RAMEN_ENCODED', 'CITY_SEATTLE_ENCODED', 'CITY_DENVER_ENCODED', 'CITY_San Mateo_encoded', 'CITY_New York City_encoded', 'CITY_BOSTON_ENCODED', 'REGION_NY_ENCODED', 'REGION_MA_ENCODED', 'REGION_CO_ENCODED', 'REGION_WA_ENCODED', 'REGION_CA_ENCODED']].drop_duplicates()
                        main_df = pd.merge(main_df, encoded_X,  how='left', left_on=['TRUCK_ID'], right_on=['TRUCK_ID']).drop_duplicates()
    
                        # Join sum data
                        sum_X = current_df[['TRUCK_ID', 'MONTH', 'HOUR', 'DAY', 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE', 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE']]
                        main_df = pd.merge(main_df, sum_X,  how='left', left_on=['TRUCK_ID', 'HOUR', 'MONTH', 'DAY'], right_on=['TRUCK_ID', 'HOUR', 'MONTH', 'DAY']).drop_duplicates()
    
                        # Fill missing values
                        main_df = main_df.fillna({
                            'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE': (main_df['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'].mean()),
                            'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE': (main_df['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'].mean())
                        })
    
                        # Write the main_df DataFrame to the "Trend_Input_Data" table in Snowflake
                        session.write_pandas(
                            df=main_df,
                            table_name="Trend_Input_Data",
                            database="FROSTBYTE_TASTY_BYTES",
                            schema="ANALYTICS",
                            quote_identifiers=True,
                            overwrite=False
                        )
    
                        # Terminate the loop if it is the last month and year
                        if year == '2022' and month == 10:
                            break
    
        except snowflake.connector.errors.ProgrammingError as e:
            print(f"Error connecting to Snowflake: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def winsorise(df, variable, upper_limit, lower_limit):
        """
        Winsorizes a numerical variable in a DataFrame by capping extreme values to specified upper and lower limits.
    
        Returns:
            pd.Series: A pandas Series containing the winsorized values of the variable.
    
        Raises:
            ValueError: If the variable is not present in the DataFrame or is not numerical.
            ValueError: If the upper_limit is less than or equal to the lower_limit.
        """
    
        # Check if the variable is present in the DataFrame and is numerical
        if variable not in df.columns or not pd.api.types.is_numeric_dtype(df[variable]):
            raise ValueError(f"The variable '{variable}' is not present in the DataFrame or is not numerical.")
    
        # Check if the upper limit is greater than the lower limit
        if upper_limit <= lower_limit:
            raise ValueError("The upper limit must be greater than the lower limit.")
    
        # Winsorize the variable using numpy where function
        return np.where(df[variable] > upper_limit, upper_limit, np.where(df[variable] < lower_limit, lower_limit, df[variable]))
    
    def generate_month_list(start_month, start_year, end_month, end_year):
        """
        Generate a list of months between the given start and end dates.
    
        Parameters:
            start_month (int): The starting month (1 to 12).
            start_year (int): The starting year.
            end_month (int): The ending month (1 to 12).
            end_year (int): The ending year.
    
        Returns:
            list: A list of month numbers (1 to 12) representing the months between the start and end dates.
        """
        try:
            start_date = datetime.date(start_year, start_month, 1)
            end_date = datetime.date(end_year, end_month, 1)
            month_list = []
    
            while start_date <= end_date:
                month_list.append(start_date.month)
                # Move to the next month by adding 32 days and then setting the day to 1
                start_date += datetime.timedelta(days=32)
                start_date = start_date.replace(day=1)
    
            return month_list
    
        except Exception as e:
            print(f"An error occurred while generating the month list: {e}")
            return []
    
    def get_shift_durations(start_hour, end_hour, num_of_locs):
        """
        Calculate the shift durations based on the starting and ending hours and the number of locations.
    
        Parameters:
            start_hour (int): The starting hour (0 to 23).
            end_hour (int): The ending hour (0 to 23).
            num_of_locs (int): The number of locations.
    
        Returns:
            list: A list of shift durations for each location.
        """
        try:
            starting_hour = start_hour
            ending_hour = end_hour
            working_hours = ending_hour - starting_hour
            # Calculate the base shift hours (without considering the remainder)
            shift_hours = working_hours // num_of_locs
            # Calculate the remaining hours to distribute
            remaining_hours = working_hours % num_of_locs
    
            # Create a list to store the shift hours for each shift
            shift_hours_list = [shift_hours] * num_of_locs
    
            # Distribute the remaining hours evenly across shifts
            for i in range(remaining_hours):
                shift_hours_list[i] += 1
    
            return shift_hours_list
    
        except Exception as e:
            print(f"An error occurred while calculating shift durations: {e}")
            return []
    
    def get_shift_hours(start_hour, end_hour, num_of_locs):
        """
        Calculate the shift hours for each shift given the starting hour, ending hour, and number of locations.
    
        Returns:
            list: A list of lists representing the shift hours for each location.
        """
        try:
            starting_hour = start_hour
            ending_hour = end_hour
            working_hours = ending_hour - starting_hour
    
            # Calculate the base shift hours (without considering the remainder)
            shift_hours = working_hours // num_of_locs
    
            # Calculate the remaining hours to distribute
            remaining_hours = working_hours % num_of_locs
    
            # Create a list to store the shift hour arrays
            shift_hours_list = []
    
            # Calculate the shift hours for each shift
            current_hour = starting_hour
            for i in range(num_of_locs):
                # Calculate the end hour for the current shift
                end_shift_hour = current_hour + shift_hours
    
                # Add the hours for the current shift to the list
                shift_hours_list.append(list(range(current_hour, end_shift_hour)))
    
                # Adjust the current hour for the next shift
                current_hour = end_shift_hour
    
                # Distribute remaining hours evenly across shifts
                if remaining_hours > 0:
                    shift_hours_list[i].append(current_hour)
                    current_hour += 1
                    remaining_hours -= 1
    
            return shift_hours_list
    
        except Exception as e:
            print(f"An error occurred while calculating shift hours: {e}")
            return []
    
    def haversine_distance(df, max_distance):
        """
        Calculate the haversine distance between two sets of latitude and longitude coordinates.
    
        Returns:
            pandas.DataFrame: A DataFrame containing the rows with distances within the maximum distance.
        """
        try:
            # Copy the input DataFrame to avoid modifying the original
            df = df.copy()
    
            # Convert latitude and longitude from degrees to radians
            df['LAT_rad'] = df['LAT'].apply(math.radians)
            df['LONG_rad'] = df['LONG'].apply(math.radians)
            df['LAT2_rad'] = df['LAT2'].apply(math.radians)
            df['LONG2_rad'] = df['LONG2'].apply(math.radians)
    
            # Haversine formula
            df['dlon'] = df['LONG2_rad'] - df['LONG_rad']
            df['dlat'] = df['LAT2_rad'] - df['LAT_rad']
            df['a'] = (df['dlat'] / 2).apply(math.sin)**2 + df['LAT_rad'].apply(math.cos) * df['LAT2_rad'].apply(math.cos) * (df['dlon'] / 2).apply(math.sin)**2
            df['c'] = 2 * df['a'].apply(lambda x: math.atan2(math.sqrt(x), math.sqrt(1 - x)))
            df['DISTANCE'] = 6371 * df['c']  # Radius of the Earth in kilometers
    
            # Filter rows based on max_distance
            df = df[df['DISTANCE'] <= max_distance]
    
            # Drop intermediate columns
            df.drop(['LAT_rad', 'LONG_rad', 'LAT2_rad', 'LONG2_rad', 'dlon', 'dlat', 'a', 'c'], axis=1, inplace=True)
    
            # Reset the index of the resulting DataFrame
            df.reset_index(drop=True, inplace=True)
    
            return df
    
        except Exception as e:
            print(f"An error occurred while calculating haversine distance: {e}")
            return pd.DataFrame()
    
    def find_distance(df1, df2):
        """
        Calculate the haversine distance between two sets of latitude and longitude coordinates.
    
        Returns:
            float: The haversine distance between the two locations in kilometers.
        """
        try:
            # Radius of the Earth in kilometers
            R = 6371
    
            lat1 = df1['LAT']
            lon1 = df1['LONG']
            lat2 = df2['LAT']
            lon2 = df2['LONG']
    
            # Convert latitude and longitude to radians
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)
    
            # Difference between latitudes and longitudes
            delta_lat = lat2_rad - lat1_rad
            delta_lon = lon2_rad - lon1_rad
    
            # Haversine formula
            a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c
    
            return distance
    
        except Exception as e:
            print(f"An error occurred while calculating the distance: {e}")
            return None
    
    def format_time_range(hours_list):
        """
        Format a list of hours into a time range string.
    
        Returns:
            str: A formatted time range string.
        """
        try:
            if len(hours_list) == 0:
                return "No hours provided"
            elif len(hours_list) == 1:
                return format_hour(hours_list[0])
            else:
                start_hour = format_hour(hours_list[0])
                end_hour = format_hour(hours_list[-1]+1)
                return f"{start_hour} to {end_hour}"
    
        except Exception as e:
            print(f"An error occurred while formatting the time range: {e}")
            return ""
    
    def format_hour(hour):
        """
        Format an hour (0 to 23) into a string representation.
    
        Returns:
            str: A formatted string representation of the hour.
        """
        try:
            if hour == 0:
                return "12am"
            elif hour < 12:
                return f"{hour}am"
            elif hour == 12:
                return "12pm"
            else:
                return f"{hour - 12}pm"
    
        except Exception as e:
            print(f"An error occurred while formatting the hour: {e}")
            return ""
    
    def number_of_months(start_month, start_year, end_month, end_year):
        """
        Calculate the number of months between two dates.
    
        Returns:
            int: The number of months between the two dates.
        """
        try:
            start_date = datetime.date(start_year, start_month, 1)
            end_date = datetime.date(end_year, end_month, 1)
    
            months_diff = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
    
            return months_diff
    
        except Exception as e:
            print(f"An error occurred while calculating the number of months: {e}")
            return 0
    
    def get_highest_predicted(df):
        """
        Get the highest predicted value for each day based on the given DataFrame.
    
        Returns:
            pandas.DataFrame: A DataFrame with the highest predicted value for each day, along with the corresponding "LOCATION_ID".
        """
        try:
            # Group by "LOCATION_ID" and "DAY" and calculate the sum of "Predicted"
            summed_df = df.groupby(["LOCATION_ID", "DAY"])["Predicted"].sum().reset_index()
    
            # Find the maximum summed predicted value for each day
            max_predicted_df = summed_df.groupby("DAY")["Predicted"].max().reset_index()
    
            # Merge with the original DataFrame to get the corresponding "LOCATION_ID"
            result_df = pd.merge(max_predicted_df, summed_df, on=["DAY", "Predicted"])
    
            return result_df
    
        except Exception as e:
            print(f"An error occurred while getting the highest predicted values: {e}")
            return pd.DataFrame()
    
    import matplotlib.ticker as ticker
    def create_monthly_sales_graph(monthly_df,total_revenue):
        """
        Create a monthly sales graph from the given DataFrame. Shows monthly and total revenue, saves graph as png,
        """
    
        try:
            # Convert value to thousands (K)
            df = monthly_df.copy()
            df['Value'] = monthly_df['Value'] / 1000
    
            # Increase the figure size before plotting
            plt.figure(figsize=(10, 6))
    
            # Plot time series line chart for monthly_df (green)
            plt.plot(df['Months'], df['Value'], color='green', label='Current Predictions')
    
            plt.xlabel('Month')
            plt.ylabel('Total Revenue (K)')
            plt.title('Monthly Sales')
            plt.xticks(rotation=45)
            plt.legend()
    
            # Format y-axis tick labels with '$' sign
            plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}K'))
    
            # Add total revenue annotation
            plt.annotate(f'Total Revenue: ${total_revenue}', xy=(0.01, 0.3), xycoords='axes fraction', fontsize=12)
    
            # Set y-axis limits
            plt.ylim(0)
    
            # Save the plot to a file (to be displayed later in Streamlit)
            plt.savefig("monthly_sales_graph.png")
            plt.close()
        except Exception as e:
            print(f"An error occurred while creating the monthly_sales graph: {e}")
    
    def create_revenue_per_hour_graph(monthly_df,num_of_months ,work_hours,work_days,original_df,previous_df):
        """
        Create a monthly sales graph from the given DataFrame. Shows monthly and average revenue per hour for predicted, original, and previous year's data, saves graph as png,
        """
    
        try:
            # Calculate the YoY Growth of predicted revenue per hour
            predicted_revenue_per_hour = monthly_df['Value'] / (num_of_months * work_hours * (2+len(work_days) * 4))
            previous_year_revenue_per_hour = previous_df['Value'] / (num_of_months * previous_df['Hours'] * previous_df['Days'])
            yoy_growth = ((predicted_revenue_per_hour - previous_year_revenue_per_hour) / previous_year_revenue_per_hour) * 100
    
            # Calculate the revenue per hour increase between predicted and original
            original_revenue_per_hour = original_df['Value'] / (num_of_months * original_df['Hours'] * original_df['Days'])
            predicted_increase = ((predicted_revenue_per_hour - original_revenue_per_hour) / original_revenue_per_hour) * 100
    
            # Calculate monthly average revenue per hour
            monthly_df['Value'] = monthly_df['Value'] / work_hours / (2+len(work_days)*4)
            original_df['Value'] = original_df['Value'] / original_df['Hours'] / original_df['Days']
            previous_df['Value'] = previous_df['Value'] / previous_df['Hours'] / previous_df['Days']
    
            # Calculate the average YoY Growth and predicted increase compared to original
            avg_yoy_growth = yoy_growth.mean()
            avg_predicted_increase = predicted_increase.mean()
    
            # Increase the figure size before plotting
            plt.figure(figsize=(10, 8))
    
            # Plot time series line chart for monthly_df (green)
            plt.plot(monthly_df['Months'], monthly_df['Value'], color='green', label='Current Predictions')
    
            # Plot time series line chart for original_df (red)
            plt.plot(original_df['Months'], original_df['Value'], color='red', label='Original Data')
    
            # Plot time series line chart for previous_df (purple)
            plt.plot(previous_df['Months'], previous_df['Value'], color='purple', label='Previous Year')
    
            plt.xlabel('Month')
            plt.ylabel('Total Revenue /hour')
            plt.title('Monthly Revenue Per Hour')
            plt.xticks(rotation=45)
            plt.legend()
    
            # Format y-axis tick labels with '$' sign
            plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
    
            # Add total revenue annotation
            og = (original_df['Value'].sum() / num_of_months).round(2)
            prev = (previous_df['Value'].sum()/ num_of_months).round(2)
            avg = (monthly_df['Value'].sum()/ num_of_months).round(2)
            
            plt.annotate(f'Average Revenue Per Hour: ${avg}/hr', xy=(0.01, 0.6), xycoords='axes fraction', fontsize=12)
            plt.annotate(f'Original Data: ${og}/hr', xy=(0.01, 0.5), xycoords='axes fraction', fontsize=12)
            plt.annotate(f'Previous Year: ${prev}/hr', xy=(0.01, 0.4), xycoords='axes fraction', fontsize=12)
    
            # Display difference between predicted and original as annotation
            plt.annotate(f'Average Predicted Increase from Original: {avg_predicted_increase:.2f}%', xy=(0.01, 0.3), xycoords='axes fraction', fontsize=12, color='blue')
    
            # Display average YoY Growth as annotation
            plt.annotate(f'Average YoY Growth: {avg_yoy_growth:.2f}%', xy=(0.01, 0.2), xycoords='axes fraction', fontsize=12, color='blue')
    
            # Set y-axis limits
            plt.ylim(0)
    
            # Save the plot to a file (to be displayed later in Streamlit)
            plt.savefig("monthly_revenue_per_hour_graph.png")
            plt.close()
        except Exception as e:
            print(f"An error occurred while creating the monthly_revenue_per_hour graph: {e}")
    
    def create_x_holdout_graph(df_predictions):
        """
        Create a scatter plot comparing predicted values against holdout values.
        """
    
        try:
            # Plot the predicted values against the holdout values
            plt.figure(figsize=(20, 10))
            plt.scatter(df_predictions['Holdout'], df_predictions['Predicted'], c='blue', label='Predicted vs Holdout')
    
            # Add a reference line
            plt.plot([df_predictions['Holdout'].min(), df_predictions['Holdout'].max()],
                     [df_predictions['Holdout'].min(), df_predictions['Holdout'].max()],
                     c='red', label='Perfect Prediction')
    
            # Set labels and title
            plt.xlabel('Holdout')
            plt.ylabel('Predicted')
            plt.title('Prediction Accuracy')
    
            # Show the legend
            plt.legend()
    
            # Save the plot to a file (to be displayed later in Streamlit)
            plt.savefig("x_holdout_graph.png")
            plt.close()
    
        except Exception as e:
            print(f"An error occurred while creating the x_holdout graph: {e}")
    
    try:
        xgb = minh_model
    except Exception as e:
            print(f"An error occurred while loading the model: {e}")
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    import numpy as np
    # Streamlit web app code
    def main():
        try:
            st.title("Food Truck Revenue Forecast Trend")
    
            # Disable button if there's a warning message
            button_disabled = False
    
            # Overview of the web app tab
            st.write("This tab presents food truck revenue trend forecasts, including optimal route predictions and revenue estimates on a daily basis for the selected time-frame. It allows users to view monthly graphs illustrating revenue and revenue/hr per month and also offers a comparison with the previous year's and original year's revenue data without the optimized routing algorithm. Additionally, users can observe the machine learning model's performance metrics and explore the feature importance.")
    
            # Input fields for truck data
            st.title('Input Section to determine optimal routing and forecast time-frame')
    
            # Dictionary to map truck details to their IDs
            truck_details_to_id = {
                "Cheeky Greek, Gyros, Denver, David Miller (27)": 27,
                "Peking Truck, Chinese, Denver, David Miller (28)": 28,
                "Peking Truck, Chinese, Seattle, Brittany Williams (43)": 43,
                "Nani's Kitchen, Indian, Seattle, Mary Sanders (44)": 44,
                "Freezing Point, Ice Cream, Boston, Brittany Williams (46)": 46,
                "Smoky BBQ, BBQ, Boston, Mary Sanders (47)": 47
            }
    
            # Predefined list for truck details
            truck_details_list = list(truck_details_to_id.keys())
    
            # Selectbox widget to choose a truck detail
            selected_truck_detail = st.selectbox("Select Food Truck (ID)", truck_details_list)
    
            # Get the corresponding truck ID using the dictionary
            truck_id = truck_details_to_id[selected_truck_detail]
    
            # Define the minimum and maximum allowed date range
            min_date = datetime.date(2020, 1, 1)
            max_date = datetime.date(2022, 10, 31)
    
            # Date range input widget to choose the forecast period
            date_range = st.date_input('Select a date range forecast (only month and year) For Jan 2020 to Oct 2022 only', (min_date, max_date))
    
            # Validate the selected date range
            if len(date_range) == 1:
                date_range = (date_range[0], date_range[0])  # Fix for handling a single date selection
    
            # Enforce the minimum 3 months date range
            if date_range[1] - date_range[0] < datetime.timedelta(days=3 * 30):
                st.warning("Please select a date range with a minimum of 3 months.")
                button_disabled = True
    
            # Ensure the range has at most 12 months
            if date_range[1] - date_range[0] > datetime.timedelta(days=12 * 30):
                st.warning("Please select a date range with a maximum of 12 months.")
                button_disabled = True
                # Adjust the range to have at most 12 months
                end_date = min(date_range[0] + datetime.timedelta(days=12 * 30), max_date)
                date_range = (date_range[0], end_date)
    
            # Limit the user from selecting dates beyond the minimum and maximum dates
            if date_range[0] < min_date:
                st.warning("Please select a start date within the allowed range.")
                button_disabled = True
                date_range = (min_date, date_range[1])
            elif date_range[1] > max_date:
                st.warning("Please select an end date within the allowed range.")
                button_disabled = True
                date_range = (date_range[0], max_date)
    
            # Extract the start and end year and month from the selected date range
            start_year = date_range[0].year
            start_month = date_range[0].month
            end_year = date_range[1].year
            end_month = date_range[1].month
    
            # Slider widget to select working hours range
            working_hours = st.slider('Select working hours (24h-Notation)', 1, 24, (8, 12))
    
            # Ensure the range has at least 2 hours
            if working_hours[1] - working_hours[0] < 2:
                st.warning("Please select a range with at least 2 hours.")
                button_disabled = True
                # Adjust the range to have at least 2 hours
                end_hour = min(working_hours[0] + 2, 23)
                working_hours = (working_hours[0], end_hour)
    
            # Extract the start and end hour from the selected working hours range
            start_hour = int(working_hours[0])
            end_hour = int(working_hours[1])
    
            # Handle special case when end hour is 24 (midnight)
            if end_hour == 24:
                end_hour = 0
    
            # Calculate total working hours
            work_hours = end_hour - start_hour
    
            # Number input widget to select the number of locations
            num_of_locs = st.number_input("Number of Locations", min_value=1, value=2, max_value=work_hours)
    
            # Validate the number of locations
            if num_of_locs > 8:
                st.warning("Please select a smaller number of locations (maximum 8)")
                button_disabled = True
                num_of_locs = 8
    
            # Number input widget to select the maximum travel distance for each location
            each_location_travel_distance = st.number_input("Each Location Max Travel Distance (km)", min_value=0, value=5, max_value=50)
    
            # Dictionary to map weekday names to integers
            weekdays_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    
            # List of weekday names
            weekdays_names = list(weekdays_dict.keys())
    
            # Multi-select widget to select work days
            selected_weekdays_names = st.multiselect("Select Work Days", weekdays_names, default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    
            # Convert selected weekday names to corresponding integer values
            work_days = [weekdays_dict[name] for name in selected_weekdays_names]
    
            # If no weekdays are selected, default to all weekdays (Monday=0, ..., Friday=4)
            if not work_days:
                st.warning("Please select at least one weekday")
                work_days = [0, 1, 2, 3, 4]
                button_disabled = True
    
            # Date input widget to select a specific date for the optimal route
            route_date = st.date_input('Select a specific date to see its optimal route', date_range[0])
    
            # Check if the selected date is within the allowed range
            if route_date < date_range[0]:
                # Display a warning message and adjust the date to the start of the date range
                st.warning("Please select a start date within the allowed range.")
                route_date = date_range[0]
                button_disabled = True
            elif route_date > date_range[1]:
                # Display a warning message and adjust the date to the end of the date range
                st.warning("Please select an end date within the allowed range.")
                route_date = date_range[1]
                button_disabled = True
            elif route_date.weekday() not in work_days:
                # Display a warning message if the selected date is not a working weekday
                st.warning("Please select a date that is during one of your selected working weekdays.")
                route_date = date_range[0]
                button_disabled = True
                
            # Display the selected truck id and date range
            st.subheader("Selected Truck ID: {}".format(truck_id))
            st.subheader("Selected date range: {} to {}".format(date_range[0].strftime("%B %Y"), date_range[1].strftime("%B %Y")))
            
        except Exception as e:
            print(f"An error occurred with the input section: {e}")
            
        try:
            # Process the inputs and display the results when the "Process Data" button is clicked
            if st.button("Forecast Data (Main)", disabled=button_disabled):
                # Calculate the maximum total travel distance based on each location's max travel distance and the number of locations
                max_total_travel_distance = each_location_travel_distance * num_of_locs
    
                # Generate the list of months within the selected date range
                months_list = generate_month_list(start_month, start_year, end_month, end_year)
    
                # Convert the selected working hours to a list of hours
                hours_list = list(range(start_hour, end_hour + 1))
    
                # Initialize variables
                year = start_year
                shift_hours_list = get_shift_hours(start_hour, end_hour, num_of_locs)
                month_value_list = []
                final_df = pd.DataFrame()
    
                # DataFrames for storing original and previous year's revenue information
                original_df = pd.DataFrame()
                previous_df = pd.DataFrame()
    
                # Retrieve sales data for the selected truck from the Snowflake database
                session.use_schema("ANALYTICS")
                query = 'Select * from "Sales_Forecast_Training_Data" WHERE TRUCK_ID = {}'.format(truck_id)
                X_final_scaled = session.sql(query).to_pandas()
                X_final_scaled.rename(columns={"Profit": "Revenue"}, inplace=True)
                sales_pred = session.sql("select * from ANALYTICS.SALES_PREDICTION").to_pandas()
                X_final_scaled = X_final_scaled.merge(sales_pred["l_w5i8_DATE"].astype(str).str[:4].rename('YEAR'), left_index=True, right_index=True)
                X_final_scaled = X_final_scaled[['Revenue', 'YEAR', 'MONTH', 'DAY', 'HOUR']]
    
                for month in months_list:
                    # Adjust the year if the date range spans multiple years
                    if start_year != end_year:
                        if month == 1:
                            year += 1
    
                    # Fetch input data for the current month, hours, and working weekdays from the Snowflake database
                    query = 'SELECT * FROM "Trend_Input_Data" WHERE TRUCK_ID = {} AND YEAR = {} AND MONTH = {} AND HOUR IN ({}) AND DOW IN ({});'.format(
                        truck_id, year, month, ', '.join(map(str, hours_list)), ', '.join(map(str, work_days)))
                    input_data = session.sql(query).to_pandas()
    
                    # Make predictions using the loaded machine learning model for the current input data
                    predict_df = input_data[['TRUCK_ID', 'MONTH', 'HOUR', 'DOW', 'DAY', 'PUBLIC_HOLIDAY', 'LAT', 'LONG', 'LOCATION_ID', 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE', 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE', 'WEATHERCODE', 'MENU_TYPE_GYROS_ENCODED', 'MENU_TYPE_CREPES_ENCODED', 'MENU_TYPE_BBQ_ENCODED', 'MENU_TYPE_SANDWICHES_ENCODED', 'MENU_TYPE_Mac & Cheese_encoded', 'MENU_TYPE_POUTINE_ENCODED', 'MENU_TYPE_ETHIOPIAN_ENCODED', 'MENU_TYPE_TACOS_ENCODED', 'MENU_TYPE_Ice Cream_encoded', 'MENU_TYPE_Hot Dogs_encoded', 'MENU_TYPE_CHINESE_ENCODED', 'MENU_TYPE_Grilled Cheese_encoded', 'MENU_TYPE_VEGETARIAN_ENCODED', 'MENU_TYPE_INDIAN_ENCODED', 'MENU_TYPE_RAMEN_ENCODED', 'CITY_SEATTLE_ENCODED', 'CITY_DENVER_ENCODED', 'CITY_San Mateo_encoded', 'CITY_New York City_encoded', 'CITY_BOSTON_ENCODED', 'REGION_NY_ENCODED', 'REGION_MA_ENCODED', 'REGION_CO_ENCODED', 'REGION_WA_ENCODED', 'REGION_CA_ENCODED']]
                    predict_df['Predicted'] = np.expm1(xgb.predict(predict_df.copy()))
    
                    # Initialize a list to store DataFrames for each shift's predicted values
                    shifts_df_list = []
                    value = 0
                    for i in range(num_of_locs):
                        # Filter data for the current shift
                        current_shift = predict_df[predict_df['HOUR'].isin(shift_hours_list[i])]
    
                        if i > 0:
                            # Merge data from the previous shift to calculate the distance between locations
                            previous_shift = pd.merge(shifts_df_list[i-1], predict_df[["LOCATION_ID", 'LAT', 'LONG']].drop_duplicates(), on=["LOCATION_ID"])
                            previous_shift.rename(columns={'LAT': 'LAT2', 'LONG': 'LONG2'}, inplace=True)
                            current_shift = pd.merge(current_shift, previous_shift[['LAT2','LONG2','DAY']], on=['DAY']).drop_duplicates()
                            current_shift = haversine_distance(current_shift, each_location_travel_distance)
    
                        # Get the highest predicted revenue for each day in the current shift
                        highest_df = get_highest_predicted(current_shift)
                        value += highest_df['Predicted'].sum()
                        shifts_df_list.append(highest_df)
                        highest_df['HOUR'] = shift_hours_list[i][0]
                        highest_df['MONTH'] = month
                        highest_df['YEAR'] = year
                        final_df = pd.concat([final_df, highest_df])
    
                    # Store the total predicted revenue value for the current month
                    month_value_list.append(value)
    
                    # Calculate original and previous year's revenue for the current month
                    og = X_final_scaled[(X_final_scaled['MONTH'] == month) & (X_final_scaled['YEAR'] == str(year))]
                    og_df = pd.DataFrame({'Value': [og['Revenue'].sum()], 'Hours': [og['HOUR'].nunique()], 'Days': [og['DAY'].nunique()], 'YEAR': [year], 'Months': [month]})
                    original_df = pd.concat([original_df, og_df])
                    prev = X_final_scaled[(X_final_scaled['MONTH'] == month) & (X_final_scaled['YEAR'] == str(year-1))]
                    pre_df = pd.DataFrame({'Value': [prev['Revenue'].sum()], 'Hours': [prev['HOUR'].nunique()], 'Days': [prev['DAY'].nunique()], 'YEAR': [year-1], 'Months': [month]})
                    previous_df = pd.concat([previous_df, pre_df])
    
                # Create DataFrame to store the monthly revenue values
                monthly_df = pd.DataFrame({
                    'Months': months_list,
                    'Value': month_value_list
                })
    
                # Convert months to three-letter abbreviations
                monthly_df['Months'] = monthly_df['Months'].apply(lambda x: calendar.month_abbr[x])
                previous_df['Months'] = previous_df['Months'].apply(lambda x: calendar.month_abbr[x])
                original_df['Months'] = original_df['Months'].apply(lambda x: calendar.month_abbr[x])
    
                # Calculate the total revenue for the selected time frame
                total_revenue = monthly_df['Value'].sum()
    
                # Calculate the number of months within the selected date range
                num_of_months = number_of_months(start_month, start_year, end_month, end_year)
    
                # Format total revenue with commas as thousands separator
                total_revenue = f'{total_revenue:,.0f}'
    
                # Generate and save the monthly sales and revenue per hour graphs to image files
                create_monthly_sales_graph(monthly_df, total_revenue)
                create_revenue_per_hour_graph(monthly_df, num_of_months, work_hours, work_days, original_df, previous_df)
    
                # Display the monthly sales graph
                st.image("monthly_sales_graph.png", use_column_width=True)
    
                # Display the monthly revenue per hour graph
                st.image("monthly_revenue_per_hour_graph.png", use_column_width=True)
    
                # Retrieve additional data for the current route date from the Snowflake database
                X_final_scaled = session.sql('Select * from "Sales_Forecast_Training_Data";').to_pandas()
                final_df = pd.merge(final_df.copy(), X_final_scaled[['LOCATION_ID', 'LAT', 'LONG']].drop_duplicates(), on=["LOCATION_ID"])
                shift_durations = get_shift_durations(start_hour, end_hour, num_of_locs)
                distance_travelled = 0
                revenue_earned = 0
    
                # Loop through each shift to display details for each location and calculate revenue
                for i in range(num_of_locs):
                    current_df = final_df[(final_df['DAY'] == route_date.day) & (final_df['MONTH'] == route_date.month) & (final_df['YEAR'] == route_date.year)].iloc[i]
                    if i == num_of_locs - 1:
                        shift_hours_list[i].append(shift_hours_list[i][-1])
                    time_range = format_time_range(shift_hours_list[i])
                    st.subheader("Shift: {}".format(str(i+1)))
                    st.write(time_range)
                    st.write('Shift Hours: ', shift_durations[i])
                    st.write('Current Location Number: ', current_df['LOCATION_ID'].round(0))
                    st.write('Predicted Revenue: ', current_df['Predicted'].round(2))
                    st.write('Predicted Revenue per hour: ', (current_df['Predicted'] / shift_durations[i]).round(2))
                    revenue_earned += current_df['Predicted']
    
                # Calculate the distance travelled and the revenue earned per kilometer
                for i in range(num_of_locs - 1):
                    distance_travelled += find_distance(final_df[(final_df['DAY'] == route_date.day) & (final_df['MONTH'] == route_date.month) & (final_df['YEAR'] == route_date.year)].iloc[i], final_df[(final_df['DAY'] == route_date.day) & (final_df['MONTH'] == route_date.month) & (final_df['YEAR'] == route_date.year)].iloc[i + 1])
    
                if distance_travelled > 0:
                    rev_dis = round(revenue_earned / distance_travelled, 2)
                else:
                    rev_dis = round(revenue_earned, 2)
    
                # Display the overall route information
                st.subheader('Overall Route')
                st.write('Maximum possible distance travelled throughout all the shifts: ', max_total_travel_distance, 'km')
                st.write('Total distance travelled: ', round(distance_travelled, 2), 'km')
                st.write('Dollars earned by km travelled: $', rev_dis, '/km')
        except Exception as e:
            print(f"An error occurred while processing the data: {e}")

        selected_model = st.selectbox("Select a ML Model to see Performance", ['Old Asg2 Model', 'Updated Asg2 Model (Fixed)', 'Improved Asg3 Group Model (Main)','Javier Model','Ayrton Model','Minh Model','Nathan Model','Vibu Model'])
        st.write("Due to streamlit's memory, loading the individual models ('Javier Model','Ayrton Model','Minh Model','Nathan Model','Vibu Model') is not possible as it causes the app to crash. Please do not select them here. To view them, go to: https://minhtab3-icp-grp3.streamlit.app/")
        try:
            # Create a button to show feature importance and performance
            if st.button('Show Model Performance'):
                
                # Load data from the Snowflake database
                session.use_schema("ANALYTICS")
                X_final_scaled = session.sql('Select * from "Sales_Forecast_Training_Data";').to_pandas()
                X_final_scaled.rename(columns={"Profit": "Revenue"}, inplace=True)

                if selected_model == 'Old Asg2 Model':
                    try:
                        model_per = old_model
                    except Exception as e:
                            print(f"An error occurred while loading the model from the file: {e}")
                    # Winsorize the target and some features to reduce the impact of outliers
                    X_final_scaled['Revenue'] = winsorise(X_final_scaled, 'Revenue', X_final_scaled['Revenue'].quantile(0.85), X_final_scaled['Revenue'].quantile(0))
                    X_final_scaled['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'] = winsorise(X_final_scaled, 'SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE', X_final_scaled['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'].quantile(0.85), X_final_scaled['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'].quantile(0))
                    X_final_scaled['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'] = winsorise(X_final_scaled, 'SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE', X_final_scaled['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'].quantile(0.8), X_final_scaled['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'].quantile(0.5))
                
                elif selected_model == 'Updated Asg2 Model (Fixed)':
                    try:
                        model_per = old_updated_model
                    except Exception as e:
                            print(f"An error occurred while loading the model from the file: {e}")
                    #Trimming Outliers for Holdout Graph
                    X_final_scaled = trim_outliers(X_final_scaled, 'Revenue')
                    outliers_IV = np.where(X_final_scaled['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'] >1.7, True, np.where(X_final_scaled['SUM_DAY_OF_WEEK_AVG_CITY_MENU_TYPE'] < -1, True, False))
                    X_final_scaled = X_final_scaled.loc[~outliers_IV]
                    outliers_IV = np.where(X_final_scaled['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'] >0.7, True, np.where(X_final_scaled['SUM_PREV_YEAR_MONTH_SALES_CITY_MENU_TYPE'] < -0.7, True, False))
                    X_final_scaled = X_final_scaled.loc[~outliers_IV]
                elif selected_model == 'Ayrton Model':
                    try:
                        model_per = ayrton_model
                    except Exception as e:
                            print(f"An error occurred while loading the model from the file: {e}")
                    #Trimming Outliers for Holdout Graph
                    X_final_scaled = trim_outliers(X_final_scaled, 'Revenue')
                elif selected_model == 'Javier Model':
                    try:
                        model_per = javier_model
                    except Exception as e:
                            print(f"An error occurred while loading the model from the file: {e}")
                    #Trimming Outliers for Holdout Graph
                    X_final_scaled = trim_outliers(X_final_scaled, 'Revenue')
                elif selected_model == 'Minh Model':
                    try:
                        model_per = minh_model
                    except Exception as e:
                            print(f"An error occurred while loading the model from the file: {e}")
                    #Trimming Outliers for Holdout Graph
                    X_final_scaled = trim_outliers(X_final_scaled, 'Revenue')
                    X_final_scaled['Revenue'] = np.log1p(X_final_scaled['Revenue'])
                elif selected_model == 'Nathan Model':
                    try:
                        model_per = nathan_model
                    except Exception as e:
                            print(f"An error occurred while loading the model from the file: {e}")
                    #Trimming Outliers for Holdout Graph
                    X_final_scaled = trim_outliers(X_final_scaled, 'Revenue')
                elif selected_model == 'Vibu Model':
                    try:
                        model_per = vibu_model
                    except Exception as e:
                            print(f"An error occurred while loading the model from the file: {e}")
                    #Trimming Outliers for Holdout Graph
                    X_final_scaled = trim_outliers(X_final_scaled, 'Revenue')
                elif selected_model == 'Improved Asg3 Group Model (Main)':
                    try:
                        model_per = model
                    except Exception as e:
                            print(f"An error occurred while loading the model from the file: {e}")
                    X_final_scaled = trim_outliers(X_final_scaled, 'Revenue')
    
                # Split the dataset into features (X) and target (y)
                X = X_final_scaled.drop("Revenue", axis=1)
                y = X_final_scaled["Revenue"]
    
                # Split the dataset into training and testing datasets
                X_training, X_holdout, y_training, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.2, random_state=42)

                try:
                    # Create a DataFrame with holdout values and predicted values
                    df_predictions = X_holdout.copy()
                    df_predictions['Holdout'] = y_holdout
                    holdout_predictions = model_per.predict(X_holdout)
                    df_predictions['Predicted'] = holdout_predictions
                    train_predictions=model_per.predict(X_train)
                    test_predictions=model_per.predict(X_test)
        
                    # Add a column for the differences
                    df_predictions['Difference'] = df_predictions['Predicted'] - df_predictions['Holdout']
        
                    # Get feature importance as a DataFrame
                    feature_importance = pd.DataFrame({'Feature': X_final_scaled.drop(columns='Revenue').columns, 'Importance': model_per.feature_importances_})
                    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
                    # Display the feature importance DataFrame
                    st.subheader('Feature Importance')
                    st.dataframe(feature_importance)
    
                    # Calculate performance metrics
                    y_true = df_predictions['Holdout']
                    y_pred = df_predictions['Predicted']
                    
                    train_mae = mean_absolute_error(y_train, train_predictions)
                    train_mse = mean_squared_error(y_train, train_predictions)
                    train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
                    if selected_model == 'Minh Model':
                        train_r2 = r2_score(np.expm1(y_train), np.expm1(train_predictions))
                    else:
                        train_r2 = r2_score(y_train, train_predictions)
                        
                    test_mae = mean_absolute_error(y_test, test_predictions)
                    test_mse = mean_squared_error(y_test, test_predictions)
                    test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
                    if selected_model == 'Minh Model':
                        test_r2 = r2_score(np.expm1(y_test), np.expm1(test_predictions))
                    else:
                        test_r2 = r2_score(y_test, test_predictions)
                        
                    mae = mean_absolute_error(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
                    if selected_model == 'Minh Model':
                        r2 = r2_score(np.expm1(y_true), np.expm1(y_pred))
                        result_df = pd.DataFrame({'True Values': np.expm1(y_true), 'Predicted Values': np.expm1(y_pred)})
                    else:
                        r2 = r2_score(y_true, y_pred)
                        result_df = pd.DataFrame({'True Values': y_true, 'Predicted Values': y_pred})
                        
                except Exception as e:
                    st.write(f"An error occurred while showing the model performance: {e}")
        
                if selected_model == 'Minh Model':
                    st.subheader('Model Performance on Training data')
                    st.write(f'Mean Absolute Error (MAE): {train_mae:.5f}')
                    st.write(f'Mean Squared Error (MSE): {train_mse:.5f}')
                    st.write(f'Root Mean Squared Error (RMSE): {train_rmse:.5f}')
                    st.write(f'R-squared (R2) score: {train_r2:.5f}')
    
                    st.subheader('Model Performance on Testing data')
                    st.write(f'Mean Absolute Error (MAE): {test_mae:.5f}')
                    st.write(f'Mean Squared Error (MSE): {test_mse:.5f}')
                    st.write(f'Root Mean Squared Error (RMSE): {test_rmse:.5f}')
                    st.write(f'R-squared (R2) score: {test_r2:.5f}')
        
                    # Display the performance metrics
                    st.subheader('Model Performance on Holdout data')
                    st.write(f'Mean Absolute Error (MAE): {mae:.5f}')
                    st.write(f'Mean Squared Error (MSE): {mse:.5f}')
                    st.write(f'Root Mean Squared Error (RMSE): {rmse:.5f}')
                    st.write(f'R-squared (R2) score: {r2:.5f}')
                else:
                    st.subheader('Model Performance on Training data')
                    st.write(f'Mean Absolute Error (MAE): {train_mae:.2f}')
                    st.write(f'Mean Squared Error (MSE): {train_mse:.2f}')
                    st.write(f'Root Mean Squared Error (RMSE): {train_rmse:.2f}')
                    st.write(f'R-squared (R2) score: {train_r2:.2f}')
    
                    st.subheader('Model Performance on Testing data')
                    st.write(f'Mean Absolute Error (MAE): {test_mae:.2f}')
                    st.write(f'Mean Squared Error (MSE): {test_mse:.2f}')
                    st.write(f'Root Mean Squared Error (RMSE): {test_rmse:.2f}')
                    st.write(f'R-squared (R2) score: {test_r2:.2f}')
        
                    # Display the performance metrics
                    st.subheader('Model Performance on Holdout data')
                    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
                    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
                    st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
                    st.write(f'R-squared (R2) score: {r2:.2f}')
    
                # Generate and save the holdout vs. predicted graph to an image file
                create_x_holdout_graph(df_predictions)
    
                # Display the holdout vs. predicted graph using the saved image file
                st.image("x_holdout_graph.png", use_column_width=True)
    
                # Display the true and predicted values in a DataFrame
                st.subheader('True vs. Predicted Values')
                st.dataframe(result_df)
        except Exception as e:
            print(f"An error occurred while showing the model performance: {e}")

    if __name__ == "__main__":
        try:
            main()
        except Exception as e:
            print(f"An error occurred with the streamlit web app: {e}")

    
        

 
    
