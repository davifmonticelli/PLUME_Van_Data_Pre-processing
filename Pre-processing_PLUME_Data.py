# This script was created to automatically pre-process P.L.U.M.E. Van data after multiple days of sampling
# Author: Davi de Ferreyro Monticelli, iREACH group (University of British Columbia)
# Date: 2023-06-17
# Version: 1.0.0

# P.L.U.M.E Dashboard currently (as of June 2023) records data from the:

#  - NOx monitor
#  - CO monitor
#  - O3 monitor
#  - 3D Sonic anemometer monitor (Wind direction, speed, vertical wind speed, and temperature)

# In addition, the following monitors are installed in the mobile laboratory but are
# not connected to the datalogger:

#  - UFP monitors (WCPC and FMPS)
#  - CO2 monitor
#  - BC monitor (microAethelometer)

# However, some data require pre-processing, because:

#  - NOx: instrument is set to send a 1 Volt signal for every 10 ppb, however
#         the maximum output is 2.5 V, meaning that if concentrations exceed
#         250 ppb, a flat line will occur in the data.
#         This setup (1V - 10ppb) can be changed in the instrument (check manual).

#  - CO:  due to limitations in the calibration process, the instrument records
#         data with a zero = -0.494 ppm. This was verified with a collocation
#         performed at Clark Drive monitoring station in 2022. Thus, all CO readings
#         must be adjusted according to the results of this collocation.

#  - CO2: the voltage signal to the datalogger is not reliable. Attempts were made to
#         properly convert the measured concentration to the voltage output, but unsuccessful.
#         Thus, as an alternative, we sample using the LI-COR 850 software and incorporate
#         the timeseries back in the P.L.U.M.E Dashboard sensor transcript.

#  - O3:  instrument is set to send a 1 Volt signal for every 10 ppb, however
#         the maximum output is 2.5 V, meaning that if concentrations exceed
#         250 ppb, a flat line will occur in the data. (But this never happens outdoors).
#         This setup (1V - 10ppb) can be changed in the instrument (check manual).

#  - UFP: after several failed attempts to understand the pulse output of the instrument and
#         why does it change for increasing steps in concentration, the WCPC was eventually
#         disconnected from the datalogger. Thus, as an alternative, we sample using the TSI software
#         and incorporate the timeseries back in the P.L.U.M.E Dashboard sensor transcript
#         + laboratory tests indicate that inlet pressures below or above 0 alter concentration readings.
#         this needs further attention before a code is implemented to fix it.

#  - BC:  we have not explored the connection of this instrument to the datalogger.
#         Thus, as an alternative, we sample using the microAeth software and IN THE FUTURE will incorporate
#         the timeseries back in the P.L.U.M.E Dashboard sensor transcript.

# For such reasons the functions below should save you time pre-processing all the P.L.U.M.E data
# before you can run the post-processing scripts such as merge.py, baseline.py, peak.py etc.
# also any other script associated with P.L.U.M.E data but currently not integrated to P.L.U.M.E Dashboard
# (e.g., AQ_and_EOI_Analysis.py script)

import csv
import pandas as pd
import numpy as np
from scipy.signal import correlate
import os
from datetime import datetime, timedelta
import sys
import io
import time
# Get the current user's username
username = os.getlogin()

########################################################################################################################
# Creating a custom class to save console printed messages in a single file:
# Essentially, the created file "XXX_console_output.txt" will have all printed messages from this script

class Tee(io.TextIOWrapper):
    def __init__(self, file, *args, **kwargs):
        self.file = file
        super().__init__(file, *args, **kwargs)

    def write(self, text):
        self.file.write(text)
        sys.__stdout__.write(text)  # Print to the console

# Specify the file path where you want to save the output
file_path_txt = f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\PREProcessing_console_output.txt'

# Open the file in write mode and create the Tee object
output_file_txt = open(file_path_txt, 'w')
tee = Tee(output_file_txt)

# Redirect the standard output to the Tee object
sys.stdout = tee

########################################################################################################################

########################################################################################################################
# Functions used throughout the script
########################################################################################################################

# Read sensor transcripts and convert to dataframe:
# Files needed: Sensor_transcript
# Function: Create a dataframe variable for each sensor transcript in folder "Files"

def read_Dash_files(date_to_run):
    # Read Sensor_Transcript_XXXX_XX_XX csv files
    date = date_to_run
    sensor_data = []
    with open(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\Sensor_Transcript_{date}.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            sensor_data.append(row)

    sensor_data = pd.DataFrame(sensor_data)
    return sensor_data

# NOx monitor data fixing:
# Files needed: Sensor_transcript + .txt file with logged concentrations (from SD card)
# Function: replace the data in Sensor transcript with the SD card data after performing a timeseries alignment

def read_NOx_files(date_to_run, dash_to_run):
    # Read NOx_SD_XXXX_XX_XX text file
    date = date_to_run
    nox_data = []
    with open(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\NOx_SD_{date}.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():  # Skip blank lines
                values = line.strip().split(',')
                nox_data.append(values)

    # Read Sensor_NOx_Transcript_XXXX_XX_XX CSV file
    sensor_data = dash_to_run

    # Create a DataFrame from the data
    nox_data = pd.DataFrame(nox_data, columns=range(1, 15))

    # Merge the date and timestamp columns and move them to the first column
    nox_data[0] = nox_data[12] + ' ' + nox_data[13]
    nox_data.drop([12, 13], axis=1, inplace=True)
    nox_data = nox_data[[0] + list(range(1, 12))]

    # Renaming columns for both datasets and dropping unecessary/incomplete rows/columns
    nox_data.columns = ['date','NO2 (ppb)','NO (ppb)','NOx (ppb)','Temp (C)','Pressure (mbar)','Flow rate (cc/min)','O3 flow rate (cc/min)','Sample voltage (volts)','O3 generator voltage (volts)','Scrubber Temp (C)','Error']
    nox_data = nox_data.drop(nox_data.index[-1]).reset_index(drop=True)  # Last row is often incomplete

    # Converting time formats
    nox_data['date'] = pd.to_datetime(nox_data['date'], format='%m/%d/%y %H:%M:%S')
    nox_data['date'] = nox_data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    nox_data['date'] = pd.to_datetime(nox_data['date'], format='%Y-%m-%d %H:%M:%S')

    # Adjust values formats: change the format of all values to float, except for the "date" column
    nox_data = nox_data.astype({col: float for col in nox_data.columns if col != 'date'})

    # Merging
    merged_data = pd.merge(nox_data, sensor_data, on='date')
    columns_to_keep = ['date','NO2 (ppb)_x','NO (ppb)_x','NO2 (ppb)_y','NO (ppb)_y']
    merged_data = merged_data[columns_to_keep]
    merged_data.columns = ['date','NO2 (ppb) SD','NO (ppb) SD','NO2 (ppb) Dash','NO (ppb) Dash']

    # Perform time series alignment using Cross Correlation Factor (CCF)
    ccf_no2 = correlate(merged_data['NO2 (ppb) SD'].astype(float), merged_data['NO2 (ppb) Dash'].astype(float))
    ccf_no = correlate(merged_data['NO (ppb) SD'].astype(float), merged_data['NO (ppb) Dash'].astype(float))

    # Find the lag that maximizes the CCF value
    lag_no2 = np.argmax(ccf_no2)
    lag_no = np.argmax(ccf_no)

    # Adjust the time series by shifting the data based on the lag
    merged_data['NO2 (ppb) SD'] = np.roll(merged_data['NO2 (ppb) SD'], -lag_no2)
    merged_data['NO (ppb) SD'] = np.roll(merged_data['NO (ppb) SD'], -lag_no)

    # Replace "NO2 (ppb)" and "NO (ppb)" when "NO2 (ppb) SD", "NO (ppb) SD", "NO2 (ppb) Dash" and "NO (ppb) Dash" are above 250 ppb

    # Select rows from merged_data where 'Dash' is equal/above to 250 and 'SD' is above 250
    # Get the specific times when the conditions are met
    specific_no2_times = merged_data.loc[(merged_data['NO2 (ppb) SD'] > 250) & (merged_data['NO2 (ppb) Dash'] > 250), 'date']
    specific_no_times = merged_data.loc[(merged_data['NO (ppb) SD'] > 250) & (merged_data['NO (ppb) Dash'] > 250), 'date']

    # Update the values of 'NO2 (ppb)' and 'NO (ppb)' in sensor_data for the specific times
    sensor_data_mod = sensor_data
    if not specific_no2_times.empty:
        sensor_data_mod.loc[sensor_data_mod['date'].isin(specific_no2_times), 'NO2 (ppb)'] = merged_data.loc[merged_data['date'].isin(specific_no2_times), 'NO2 (ppb) SD'].values

    if not specific_no_times.empty:
        sensor_data_mod.loc[sensor_data_mod['date'].isin(specific_no_times), 'NO (ppb)'] = merged_data.loc[merged_data['date'].isin(specific_no_times), 'NO (ppb) SD'].values

    return sensor_data_mod, merged_data

# CO2 monitor data integration:
# Files needed: Sensor_transcript + LICOR Software output file
# Function: replace the values in Sensor_transcript according to the timestamp (clock is the same as PC, no alignment needed)

def read_CO2_files(date_to_run, dash_to_run):
    date = date_to_run
    # Read the LICOR txt file
    co2_data = pd.read_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\CO2_LICOR_{date}.txt', delimiter='\t', skiprows=1)

    # Create a DataFrame from the sensor data
    sensor_data = dash_to_run

    # Function to merge the text in the first two columns
    def merge_text(row):
        return row['System_Date_(Y-M-D)'] + ' ' + row['System_Time_(h:m:s)']

    # Apply the 'merge_text' function to create a new merged column
    last_column = co2_data.apply(merge_text, axis=1)
    # Delete the first two columns
    co2_data = co2_data.iloc[:, 2:]
    # Insert the last column at the first position
    co2_data.insert(0, 'date', last_column)

    # Renaming columns for both datasets and dropping unecessary/incomplete rows/columns
    co2_data = co2_data.iloc[:, :2]  # Keep only first 2 columns
    co2_data.columns = ['date', 'CO2 (ppm)']

    # Converting time formats
    co2_data['date'] = pd.to_datetime(co2_data['date'], format='%Y-%m-%d %H:%M:%S')

    # Adjust values formats: change the format of all values to float, except for the "date" column (and Error column)
    co2_data = co2_data.astype({col: float for col in co2_data.columns if col != 'date'})

    # Merging
    merged_data = pd.merge(co2_data, sensor_data, on='date', how='inner')
    # Update the values
    merged_data['CO2 (ppm)_y'] = merged_data['CO2 (ppm)_x']
    # Organize it
    columns_to_keep = ['date', 'NO2 (ppb)', 'UFP (#/cm^3)', 'O3 (ppb)', 'CO (ppm)', 'CO2 (ppm)_y', 'NO (ppb)',
                       'WS (m/s)', 'WD (degrees)', 'WV (m/s)']
    merged_data = merged_data[columns_to_keep]
    merged_data.columns = ['date', 'NO2 (ppb)', 'UFP (#/cm^3)', 'O3 (ppb)', 'CO (ppm)', 'CO2 (ppm)', 'NO (ppb)',
                           'WS (m/s)', 'WD (degrees)', 'WV (m/s)']

    sensor_data_mod = merged_data

    return sensor_data_mod

# O3 monitor data fixing:
# Files needed: Sensor_transcript + .txt file with logged concentrations (from SD card)
# Function: replace the data in Sensor transcript with the SD card data after performing a timeseries alignment

def read_O3_files(date_to_run, dash_to_run):
    # Read O3_SD_XXXX_XX_XX text file
    date = date_to_run
    o3_data = []
    with open(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\O3_SD_{date}.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():  # Skip blank lines
                values = line.strip().split(',')
                o3_data.append(values)

    # Read Sensor_O3_Transcript_08_08_2022 CSV file
    sensor_data = dash_to_run

    # Create a DataFrame from the data
    o3_data = pd.DataFrame(o3_data, columns=range(0, 6))

    # Merge the date and timestamp columns and move them to the first column
    o3_data[6] = o3_data[4] + ' ' + o3_data[5]
    o3_data.drop([4, 5], axis=1, inplace=True)
    o3_data = o3_data[[6] + list(range(0, 4))]

    # Renaming columns for both datasets and dropping unecessary/incomplete rows/columns
    o3_data.columns = ['date','O3 (ppb)','Temp (C)','Pressure (mbar)','Flow rate (cc/min)']
    o3_data = o3_data.drop(o3_data.index[-1]).reset_index(drop=True)  # Last row is often incomplete

    # Converting time formats
    o3_data['date'] = pd.to_datetime(o3_data['date'], format='%m/%d/%y %H:%M:%S')
    o3_data['date'] = o3_data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    o3_data['date'] = pd.to_datetime(o3_data['date'], format='%Y-%m-%d %H:%M:%S')

    # Adjust values formats: change the format of all values to float, except for the "date" column
    o3_data = o3_data.astype({col: float for col in o3_data.columns if col != 'date'})

    # Merging
    merged_data = pd.merge(o3_data, sensor_data, on='date')
    columns_to_keep = ['date','O3 (ppb)_x','O3 (ppb)_y']
    merged_data = merged_data[columns_to_keep]
    merged_data.columns = ['date','O3 (ppb) SD','O3 (ppb) Dash']

    # Perform time series alignment using Cross Correlation Factor (CCF)
    ccf_o3 = correlate(merged_data['O3 (ppb) SD'].astype(float), merged_data['O3 (ppb) Dash'].astype(float))

    # Find the lag that maximizes the CCF value
    lag_o3 = np.argmax(ccf_o3)

    # Adjust the time series by shifting the data based on the lag
    merged_data['O3 (ppb) SD'] = np.roll(merged_data['O3 (ppb) SD'], -lag_o3)

    # Replace "O3 (ppb)" when "O3 (ppb) SD" and "O3 (ppb) Dash" are both above 250 ppb

    # Select rows from merged_data where 'Dash' is equal/above to 250 and 'SD' is above 250
    # Get the specific times when the conditions are met
    specific_o3_times = merged_data.loc[(merged_data['O3 (ppb) SD'] > 250) & (merged_data['O3 (ppb) Dash'] > 250), 'date']

    # Update the values of 'O3 (ppb)' in sensor_data for the specific times
    sensor_data_mod = sensor_data
    if not specific_o3_times.empty:
        sensor_data_mod.loc[sensor_data_mod['date'].isin(specific_o3_times), 'O3 (ppb)'] = merged_data.loc[merged_data['date'].isin(specific_o3_times), 'O3 (ppb) SD'].values

    return sensor_data_mod, merged_data

# UFP monitor data integration:
# Files needed: Sensor_transcript + TSI Software output file
# Function: replace the values in Sensor_transcript according to the timestamp (clock is the same as PC, no alignment needed)

def read_UFP_files(date_to_run, dash_to_run):
    date = date_to_run

    # Read the TSI WCPC CSV file and skip the first 16 lines (always check file to confirm # of lines to skip)
    ufp_data = pd.read_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\UFP_WCPC_{date}.csv', skiprows=16)

    # Create a DataFrame from the sensor data
    sensor_data = dash_to_run

    ufp_data = ufp_data.iloc[:, :5]  # Keep only first 5 columns
    ufp_data.columns = ['date', 'Elapsed time (s)', 'UFP (#/cm^3)', 'Counts', 'Error']

    # Converting time formats
    ufp_data['date'] = pd.to_datetime(ufp_data['date'], format='%Y-%m-%d %H:%M:%S')

    # Adjust values formats: change the format of all values to float, except for the "date" column (and Error column)
    ufp_data = ufp_data.astype({col: float for col in ufp_data.columns if col not in ['date', 'Error']})

    # Merging by time
    merged_data = pd.merge(ufp_data, sensor_data, on='date', how='inner')
    # Update the values
    merged_data['UFP (#/cm^3)_y'] = merged_data['UFP (#/cm^3)_x']
    # Organize it
    columns_to_keep = ['date', 'NO2 (ppb)', 'UFP (#/cm^3)_y', 'O3 (ppb)', 'CO (ppm)', 'CO2 (ppm)', 'NO (ppb)',
                       'WS (m/s)', 'WD (degrees)', 'WV (m/s)']
    merged_data = merged_data[columns_to_keep]
    merged_data.columns = ['date', 'NO2 (ppb)', 'UFP (#/cm^3)', 'O3 (ppb)', 'CO (ppm)', 'CO2 (ppm)', 'NO (ppb)',
                           'WS (m/s)', 'WD (degrees)', 'WV (m/s)']

    sensor_data_mod = merged_data

    return sensor_data_mod

# CO monitor data fixing:
# Files needed: Sensor_transcript
# Function: apply the 2022 collocation calibration (equation)

def read_CO_files(dash_to_run):

    # Create a DataFrame from the data
    sensor_data = dash_to_run

    # Apply equation to fix CO (ppm) data in sensor_data
    # Equation: y=0.6036x + 0.5676 (R2 = 0.45)

    # Define the equation or function to fix the data
    def fix_co_value(value):
        # Apply your equation or logic to fix the value
        fixed_value = (value*0.6036)+0.5676  # Replace with your new equation if calibration is performed again

        return fixed_value

    # Apply the fix_co_value function to the "co (ppm)" column using apply()
    sensor_data_mod = sensor_data
    sensor_data_mod['CO (ppm)'] = sensor_data_mod['CO (ppm)'].apply(lambda x: fix_co_value(x))

    return sensor_data_mod

# BC monitor data integration:
# Files needed: Sensor_transcript + microAeth Software output file
# Function: replace the values in Sensor_transcript according to the timestamp (clock is the same as PC, no alignment needed)

# (...)

# Function to read .csv files where there is a line skip
# (sometimes this happens for the Events marker files)
def read_csv_skip_blank(filename):
    rows = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line and "NA" not in line:  # Skip blank lines and lines containing "NA"
                values = line.split(',')[:13]
                rows.append(values)


    df = pd.DataFrame(rows[1:], columns=rows[0][:13])
    print(df)
    return df

# Function to read Event Markers .csv files
def read_event_marker(date_to_run):
    # Read Event_Markers_XXXX_XX_XX csv files
    date = date_to_run
    events_data = read_csv_skip_blank(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\renamed\\Event_Markers_{date}.csv')

    # Filter rows where the value in 'Type' is equal to 'manual'
    desired_value = 'manual'
    events_data = events_data[events_data['Type'] == desired_value]

    # List columns to keep
    columns_to_keep = ["Event Tag", "Time"]
    # Get a list of all columns in the original DataFrame
    all_columns = events_data.columns.tolist()
    # Create a list of columns to drop (all except specified columns)
    columns_to_drop = [column for column in all_columns if column not in columns_to_keep]
    # Drop the columns from the original DataFrame
    events_data.drop(columns_to_drop, axis=1, inplace=True)

    # Organizing it
    new_column_names = {'Event Tag': 'EOI', 'Time': 'date'}
    events_data.rename(columns=new_column_names, inplace=True)
    events_data = events_data[['date', 'EOI']]

    # Drop rows with all NaN or missing values (blank rows)
    events_data.dropna(axis=0, how='all', inplace=True)

    return events_data


def read_FMPS_files(date_to_run):
    date = date_to_run
    # Step 1: Read the CSV file
    df = pd.read_csv(
        f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\originals\\FMPS_Corrected_{date}.csv',
        header=None, low_memory=False)
    # Step 2: Store the time found in the second column of the first row in a variable
    current_date = df.iloc[0, 1]
    #current_date = datetime.strptime(current_date, "%Y-%m-%d %H:%M")
    try:
        current_date = datetime.strptime(current_date, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Handle the case when seconds are not present in the date
        current_date = datetime.strptime(current_date, "%Y-%m-%d %H:%M")
    # Step 3: Set line 13 as the new column names
    new_column_names = df.iloc[12].tolist()
    # Step 4: Skip the first 14 lines
    df = df[14:]
    df = df.reset_index(drop=True)
    new_column_names = [str(column) for column in new_column_names]
    df.columns = new_column_names
    df.rename(columns={'Channel Size [nm]:': 'timestamp'}, inplace=True)
    # Step 5: Fix the time component (column timestamp)
    # Initialize a list to store the combined date and timestamp values
    combined_datetime = []
    # Iterate through the "timestamp" column and append dates
    for timestamp_str in df['timestamp']:
        # print("timestamp string is: ", timestamp_str)
        timestamp = datetime.strptime(timestamp_str, '%I:%M:%S %p').time()
        combined_datetime.append(current_date.replace(hour=timestamp.hour, minute=timestamp.minute, second=timestamp.second))
        new_datetime = current_date.replace(hour=timestamp.hour, minute=timestamp.minute, second=timestamp.second)

        if new_datetime.time() == datetime.strptime('11:59:00 PM', '%I:%M:%S %p').time():
            current_date += timedelta(days=1)  # Increment the date by one day
    # Add the combined datetime values to the DataFrame
    df['timestamp'] = combined_datetime
    # Step 6: Add "D" in front of every string in column names except the first column
    df.columns = ['timestamp' if col == 'timestamp' else 'D' + col for col in df.columns]
    # Step 7: Delete any column after column 33
    if len(df.columns) > 32:
        df = df.iloc[:, :33]
    # Step 8: Convert all values to float except the first column
    for column in df.columns[1:]:
        df[column] = df[column].astype(float)
    # Step 9: Divide all values in the DataFrame by 16, except the first column
    df.iloc[:, 1:] = df.iloc[:, 1:] / 16

    return df


def adjust_FMPS_data(df): # Without correction for WCPC data
    mid_point = ["D6.04", "D6.98", "D8.06", "D9.31", "D10.8", "D12.4", "D14.3", "D16.5", "D19.1", "D22.1", "D25.5", "D29.4", "D34.0", "D39.2", "D45.3", "D52.3", "D60.4", "D69.8", "D80.6", "D93.1", "D107.5", "D124.1", "D143.3", "D165.5", "D191.1", "D220.7", "D254.8", "D294.3", "D339.8", "D392.4", "D453.2", "D523.3"]
    slope = [1, 1, 1.362, 0.820, 0.835, 1.139, 1.294, 1.227, 1.193, 1.215, 1.134, 0.951, 0.885, 0.935, 0.924, 0.928, 0.904, 0.913, 0.930, 0.936, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    intercept = [0, 0, 144, 19, 44, 103, 205, 443, 816, 1324, 1070, 624, 463, 258, 266, 240, 278, 126, 86, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Create Zimmerman_Table_3 DataFrame
    zimmerman_table_3 = pd.DataFrame({'mid_point': mid_point, 'slope': slope, 'intercept': intercept})

    # Get the slope values from Zimmerman_Table_3 and store them in a dictionary
    slopes = dict(zip(zimmerman_table_3['mid_point'], zimmerman_table_3['slope']))

    # Get the intercept values from Zimmerman_Table_3 and store them in a dictionary
    intercepts = dict(zip(zimmerman_table_3['mid_point'], zimmerman_table_3['intercept']))

    adjusted_df = pd.DataFrame()

    for col in df.columns[1:]:
        adjusted_df = df
        # Apply the adjustment only to non-zero values
        non_zero_mask = df[col] != 0
        adjusted_df.loc[non_zero_mask, col] = (df.loc[non_zero_mask, col] * slopes[col] + intercepts[col])

    return adjusted_df

###################################################################################################################
# MAIN
###################################################################################################################

start_time = time.time()
start_time_formatted = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
print("Start time of run: ", start_time_formatted)

import re
import shutil

# File renaming:
# Files needed: Sensor_transcript + .txt or .csv files with logged concentrations (from SD cards, LI-COR, and WCPC)
# Function: rename all files in the folder to match the expected format ("Something_YYYY_MM_DD") to run the script

# Specify the path to the folder containing the files
folder_path = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\originals"

# Get a list of file names in the folder
file_names = os.listdir(folder_path)
old_files = file_names
print("Old file names are:", file_names)

# Iterate over the file names and rename each file
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    destine_path = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\renamed"

    # Rename "Sensor Transcript YYYY-MM-DD.csv" files
    if file_name.startswith("Sensor Transcript"):
        date_str = file_name.split(" ")[-1].replace(".csv", "")
        date_str = date_str.replace("-", "_")
        new_file_name = f"Sensor_Transcript_{date_str}.csv"
        new_file_path = os.path.join(destine_path, new_file_name)
        shutil.copy(file_path, new_file_path)

    # Rename "Event Markers YYYY-MM-DD.csv" files
    if file_name.startswith("Event Markers"):
        date_str = file_name.split(" ")[-1].replace(".csv", "")
        date_str = date_str.replace("-", "_")
        new_file_name = f"Event_Markers_{date_str}.csv"
        new_file_path = os.path.join(destine_path, new_file_name)
        shutil.copy(file_path, new_file_path)

    # Rename "DD-MM-YYYY.txt" and "LOGXX.txt" files
    elif file_name.endswith(".txt"):
        if len(file_name) > 10:  # Likely a CO2 data file "DD-MM-YYYY.txt" (14 characters) vs. LOGXX.txt (9 chr.)
            with open(file_path, "r") as file:
                date_str = file_name.replace(".txt", "")
                parts = date_str.split("-")
                new_file_name = f"CO2_LICOR_{parts[2]}_{parts[1]}_{parts[0]}.txt"
                new_file_path = os.path.join(destine_path, new_file_name)
                shutil.copy(file_path, new_file_path)
        else:
            with open(file_path, "r") as file:
                columns = file.readline().strip().split(',')
                if len(columns) == 6:  # O3 data record
                    day, month, year = columns[4].split("/")  # date column is 5
                    new_file_name = f"O3_SD_20{year}_{month}_{day}.txt"
                    new_file_path = os.path.join(destine_path, new_file_name)
                    shutil.copy(file_path, new_file_path)
                elif len(columns) == 14:  # NOx data record
                    day, month, year = columns[11].split("/")  # date column is 12
                    new_file_name = f"NOx_SD_20{year}_{month}_{day}.txt"
                    new_file_path = os.path.join(destine_path, new_file_name)
                    shutil.copy(file_path, new_file_path)

    # Rename "YYYY-MM-DD HHmmSS_1 Hz.csv" files (WCPC)
    elif file_name.endswith("Hz.csv"):
        date_str = file_name.split(" ")[0]
        date_str = date_str.replace("-", "_")
        new_file_name = f"UFP_WCPC_{date_str}.csv"
        new_file_path = os.path.join(destine_path, new_file_name)
        shutil.copy(file_path, new_file_path)

print("Files renamed successfully.")

# Find Sensor_transcript and AQ monitors dataframes in the folder based on their name
file_names = os.listdir(f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\renamed")
new_files = [item for item in file_names if item not in old_files]
print("New file names are:", file_names)

# Copy files so globals() work (super annoying...)
source_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\renamed"
destination_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files"
file_to_delete_after_list = []

# Iterate over each file in the source folder
for file_name in os.listdir(source_folder):
    source_file = os.path.join(source_folder, file_name)
    destination_file = os.path.join(destination_folder, file_name)

    # Copy the file to the destination folder
    shutil.copy(source_file, destination_file)

    # Add the file name to the list
    file_to_delete_after_list.append(file_name)

# Sensor transcript data
Sensor_names = [file_name for file_name in file_names if re.match(r"Sensor_Transcript_\d{4}_\d{2}_\d{2}.csv", file_name)]
Sensor_names = [os.path.splitext(file_name)[0] for file_name in Sensor_names]  # Get rid of the .csv
# NOx SD data
NOx_names = [file_name for file_name in file_names if re.match(r"NOx_SD_\d{4}_\d{2}_\d{2}.txt", file_name)]
NOx_names = [os.path.splitext(file_name)[0] for file_name in NOx_names]  # Get rid of the .txt
# O3 SD data
O3_names = [file_name for file_name in file_names if re.match(r"O3_SD_\d{4}_\d{2}_\d{2}.txt", file_name)]
O3_names = [os.path.splitext(file_name)[0] for file_name in O3_names]  # Get rid of the .txt
# CO2 data
CO2_names = [file_name for file_name in file_names if re.match(r"CO2_LICOR_\d{4}_\d{2}_\d{2}.txt", file_name)]
CO2_names = [os.path.splitext(file_name)[0] for file_name in CO2_names]  # Get rid of the .txt
# UFP data
UFP_names = [file_name for file_name in file_names if re.match(r"UFP_WCPC_\d{4}_\d{2}_\d{2}.csv", file_name)]
UFP_names = [os.path.splitext(file_name)[0] for file_name in UFP_names]  # Get rid of the .csv

print("Dashboard names are:", Sensor_names)
print("NOx names are:", NOx_names)
print("O3 names are:", O3_names)
print("CO2 names are:", CO2_names)
print("UFP names are:", UFP_names)

# Get the unique dates from the Sensor_transcript and AQ monitors dataframe names
Sensor_dates = list(set([re.sub(r"Sensor_Transcript_", "", var_name) for var_name in Sensor_names]))
NOx_dates = list(set([re.sub(r"NOx_SD_", "", var_name) for var_name in NOx_names]))
O3_dates = list(set([re.sub(r"O3_SD_", "", var_name) for var_name in O3_names]))
CO2_dates = list(set([re.sub(r"CO2_LICOR_", "", var_name) for var_name in CO2_names]))
UFP_dates = list(set([re.sub(r"UFP_WCPC_", "", var_name) for var_name in UFP_names]))

print("Sensor dates are:", Sensor_dates)
print("NOx dates are:", NOx_dates)
print("O3 dates are:", O3_dates)
print("CO2 dates are:", CO2_dates)
print("UFP dates are:", UFP_dates)

# Find the dates that are common to both Sensor (Dash) and AQ monitors dataframes
common_NOX_dates = list(set(Sensor_dates) & set(NOx_dates))
common_O3_dates = list(set(Sensor_dates) & set(O3_dates))
common_CO2_dates = list(set(Sensor_dates) & set(CO2_dates))
common_UFP_dates = list(set(Sensor_dates) & set(UFP_dates))

print("Sensor + NOx common dates are:", common_NOX_dates)
print("Sensor + O3 common dates are:", common_O3_dates)
print("Sensor + CO2 common dates are:", common_CO2_dates)
print("Sensor + UFP common dates are:", common_UFP_dates)
print("Processing all...")
print("")

# Read Dash files and create proper dataframes:
for date in Sensor_dates:
    print("Processing Dash file w/ date: ", date)
    globals()["Sensor_Transcript" + date] = read_Dash_files(date)
    # Rename columns
    globals()["Sensor_Transcript" + date].columns = ['row', 'date', 'NO2 (ppb)', 'UFP (#/cm^3)', 'O3 (ppb)', 'CO (ppm)', 'CO2 (ppm)', 'NO (ppb)', 'WS (m/s)', 'WD (degrees)', 'WV (m/s)']
    # First row is the column names
    globals()["Sensor_Transcript" + date] = globals()["Sensor_Transcript" + date].drop(globals()["Sensor_Transcript" + date].index[0]).reset_index(drop=True)
    # Column 'row' is not necessary
    globals()["Sensor_Transcript" + date] = globals()["Sensor_Transcript" + date].drop(globals()["Sensor_Transcript" + date].columns[0],axis=1)
    # Adjust time format to datetime
    globals()["Sensor_Transcript" + date]['date'] = pd.to_datetime(globals()["Sensor_Transcript" + date]['date'], format='%Y-%m-%d %H:%M:%S')
    # Adjust data format to float
    globals()["Sensor_Transcript" + date] = globals()["Sensor_Transcript" + date].astype({col: float for col in globals()["Sensor_Transcript" + date].columns if col != 'date'})
    # Checking
    print("Dashboard output for date ", date, " is:")
    print(globals()["Sensor_Transcript" + date])

# Modify the Dashboard output for dates with NOx data from SD cards:
for date in common_NOX_dates:
    print("Processing NOx SD file w/ date: ", date)
    globals()["Sensor_Transcript" + date], globals()["Merged_Dash_NOxSD_Transcript" + date] = read_NOx_files(date, globals()["Sensor_Transcript" + date])
    globals()["Merged_Dash_NOxSD_Transcript" + date].to_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\Merged_Dash_NOxSD_Transcript_{date}.csv', index=False)

# Modify the Dashboard output for dates with O3 data from SD cards:
for date in common_O3_dates:
    print("Processing O3 SD file w/ date: ", date)
    globals()["Sensor_Transcript" + date], globals()["Merged_Dash_O3SD_Transcript" + date] = read_O3_files(date, globals()["Sensor_Transcript" + date])
    globals()["Merged_Dash_O3SD_Transcript" + date].to_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\Merged_Dash_O3SD_Transcript_{date}.csv', index=False)

# Modify the Dashboard output for dates with UFP data from TSI WCPC software exports:
for date in common_UFP_dates:
    print("Processing WCPC file w/ date: ", date)
    globals()["Sensor_Transcript" + date] = read_UFP_files(date, globals()["Sensor_Transcript" + date])

# Modify the Dashboard output for dates with CO2 data from LICOR software exports:
for date in common_CO2_dates:
    print("Processing LICOR file w/ date: ", date)
    globals()["Sensor_Transcript" + date] = read_CO2_files(date, globals()["Sensor_Transcript" + date])

# Modify the Dashboard output to account the collocation+calibration of the CO sensor:
for date in Sensor_dates:
    print("Processing Dash file w/ date: ", date)
    globals()["Sensor_Transcript" + date] = read_CO_files(globals()["Sensor_Transcript" + date])

# Save all the new Dashboard outputs:
for date in Sensor_dates:
    print("Saving new Dashboard file for date: ", date)
    globals()["Sensor_Transcript" + date].to_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\Sensor_Transcript_{date}_UPDATED.csv', index=False)

# Cleaning:
folder_path = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files"

# Iterate over each file in previously composed list and delete it
for file_name in file_to_delete_after_list:
    file_path = os.path.join(folder_path, file_name)
    os.remove(file_path)

########################################################################################################################
# Extras:
########################################################################################################################

# Processing Event Markers:

# Acquiring data
Events_names = [file_name for file_name in file_names if re.match(r"Event_Markers_\d{4}_\d{2}_\d{2}.csv", file_name)]
Events_names = [os.path.splitext(file_name)[0] for file_name in Events_names]  # Get rid of the .csv
# Get the unique dates from the EM dataframe names
Events_dates = list(set([re.sub(r"Event_Markers_", "", var_name) for var_name in Events_names]))
#print('Processing events marked at: ', Events_dates)
print("")

# Create empty dataframe
merged_df = pd.DataFrame()

for date in Events_dates:
    print('Processing events marked at: ', date)
    globals()["Event_Markers" + date] = read_event_marker(date)
    globals()["Event_Markers" + date].to_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\Event_Markers_{date}_UPDATED.csv', index=False)
    # Fill previously empty dataframe created
    merged_df = pd.concat([merged_df, globals()["Event_Markers" + date]], ignore_index=True)

merged_df.to_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\PLUME_EOIMap_All_Days.csv', index=False)

# Processing FMPS data:

# Acquiring data
FMPS_names = [file_name for file_name in file_names if re.match(r"FMPS_Corrected_\d{4}_\d{2}_\d{2}.csv", file_name)]
FMPS_names = [os.path.splitext(file_name)[0] for file_name in FMPS_names]  # Get rid of the .csv
# Get the unique dates from the Sensor_transcript and AQ monitors dataframe names
FMPS_dates = list(set([re.sub(r"FMPS_Corrected_", "", var_name) for var_name in Events_names]))
#print('Processing UFP Distribution at: ', Events_dates)
print("")

for date in Events_dates:
    print('Processing FMPS files created at: ', date)
    globals()["FMPS_Corrected_" + date] = read_FMPS_files(date)
    globals()["FMPS_Corrected_" + date] = adjust_FMPS_data(date)
    globals()["FMPS_Corrected_" + date].to_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\FMPS_Corrected_{date}_UPDATED.csv', index=False)


# Ending remarks:
print("")
end_time = time.time()
end_time_formatted = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
print("End time is: ", end_time_formatted)
print("Elapsed time is %.2f seconds." % round(end_time - start_time, 2))

# Restore the standard output
sys.stdout = sys.__stdout__

# Close the output file
output_file_txt.close()

