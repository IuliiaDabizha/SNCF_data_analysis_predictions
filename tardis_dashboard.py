#!/usr/bin/env python3
"""
** EPITECH PROJECT, 2024
** G-AIA-210-LYN-2-1-tardis
** File description:
** Main entry point for data analysis
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np


# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # Coerce bad date strings
    df = df.dropna(subset=["Date"])  # Remove rows with invalid dates

    # Normalize column names (strip spaces) and rename delay column
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={"Average delay of all trains at departure": "delay_minutes"}
    )

    return df


@st.cache_resource
def load_model():
    # Load the trained model and the fitted OneHotEncoder
    model = joblib.load("best_delay_predictor.joblib")
    encoder = joblib.load("one_hot_encoder.joblib")
    # We also need the list of feature names the model was trained on
    # This is not saved, so we'll need to recreate X_final column names
    # based on the encoder's fitted features and original numerical features.
    # A better approach would be to save the trained column list or the full X_final
    # Let's try to rebuild the expected column list based on the encoder
    # and the known numerical features from train_and_evaluate.py
    numerical_features_trained = [
        "Hour",
        "DayOfWeek",
        "Month",
        "Number of scheduled trains",
        "Number of cancelled trains",
    ]
    # Assuming the percentage columns are also numerical and were included
    # Need to dynamically get these from the dataset or save the list
    # Let's read a sample of the cleaned data to get the columns
    try:
        sample_df = pd.read_csv("cleaned_dataset.csv", nrows=1)
        percentage_features_trained = [
            col for col in sample_df.columns if "pct delay" in col.lower()
        ]
        all_numerical_features = (
            numerical_features_trained + percentage_features_trained
        )
    except FileNotFoundError:
        st.error(
            "Error: cleaned_dataset.csv not found. Cannot determine all numerical features."
        )
        all_numerical_features = numerical_features_trained  # Fallback

    # Get the feature names from the fitted encoder
    categorical_feature_names_trained = encoder.get_feature_names_out()

    # Combine numerical and one-hot encoded categorical feature names
    # The order needs to match the training data (numerical first, then categorical)
    expected_features = all_numerical_features + list(categorical_feature_names_trained)

    return model, encoder, expected_features


# Update the loading call
data = load_data()
# Unpack the returned values from load_model
model, ohe, expected_features = load_model()

# Dashboard Title
st.title("🚆 SNCF Train Delay Dashboard")

st.markdown(
    """
Explore delay patterns across the French railway network and predict train delays in real-time.
Use the sidebar to filter data and get personalized insights.
"""
)

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
station = st.sidebar.selectbox(
    "Departure Station", sorted(data["Departure station"].dropna().astype(str).unique())
)

# Hardcoded list of specific dates to allow
allowed_dates_str = [
    "2018-01-01",
    "2018-02-01",
    "2018-03-01",
    "2018-04-01",
    "2018-05-01",
    "2018-06-01",
    "2018-07-01",
    "2018-08-01",
    "2018-09-01",
    "2018-10-01",
]

# Convert strings to date objects for consistent comparison and display
allowed_dates = [pd.to_datetime(d).date() for d in allowed_dates_str]

# Use a selectbox for date selection, populated with the allowed dates
if allowed_dates:
    date = st.sidebar.selectbox("Select a Date", allowed_dates)
else:
    st.sidebar.error("No allowed dates are configured.")
    date = None  # Set date to None if no allowed dates

filtered_data = data[
    (data["Departure station"] == station)
    & (data["Date"].dt.date == date)  # Compare dates directly since date is already a date object
]

# Cancellation Rate (assuming a column exists)
if "canceled" in filtered_data.columns and not filtered_data.empty:
    st.subheader("🚫 Cancellation Rate")
    cancellation_rate = filtered_data["canceled"].mean() * 100
    st.metric("Cancellation Rate", f"{cancellation_rate:.2f}%")

# Arrange plots in a grid
st.header("📊 Visualizations")

# First row of plots
col1, col2 = st.columns(2)

with col1:
    # Delay Distribution
    st.subheader("Delay Distribution at Selected Station")
    if not filtered_data.empty:
        fig1, ax1 = plt.subplots()
        sns.histplot(filtered_data["delay_minutes"], bins=20, kde=True, ax=ax1)
        ax1.set_xlabel("Delay (minutes)")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)
    else:
        st.warning(
            "No data available for the selected station and date for distribution plot."
        )


with col2:
    # Boxplot
    st.subheader("Boxplot of Delays by Station")
    fig2, ax2 = plt.subplots(figsize=(15, 7))
    sns.boxplot(data=data, x="Departure station", y="delay_minutes", ax=ax2)
    plt.setp(ax2.get_xticklabels(), rotation=90)
    plt.tight_layout()
    st.pyplot(fig2)

# Second row of plots
col3, col4 = st.columns(2)

with col3:
    # Average Delay by Day of the Week
    st.subheader("🗓️ Average Delay by Day of the Week")

    # Drop rows with NaT in 'Date' or NaN in 'delay_minutes' after coercion and create a copy
    data_cleaned_for_day_plot = data.dropna(subset=["Date", "delay_minutes"]).copy()

    if not data_cleaned_for_day_plot.empty:
        # Extract day of week (Monday=0, Sunday=6) - Modifying the copy
        data_cleaned_for_day_plot["DayOfWeek"] = data_cleaned_for_day_plot[
            "Date"
        ].dt.dayofweek

        # Map dayofweek number to name for better visualization - Modifying the copy
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        data_cleaned_for_day_plot["DayOfWeekName"] = data_cleaned_for_day_plot[
            "DayOfWeek"
        ].map(lambda x: day_names[x])

        # Calculate average delay by day of week and sort by day order
        average_delay_by_day = (
            data_cleaned_for_day_plot.groupby("DayOfWeekName")["delay_minutes"]
            .mean()
            .reindex(day_names)
        )

        fig4, ax4 = plt.subplots()
        # Address FutureWarning by setting hue and legend
        sns.barplot(
            x=average_delay_by_day.index,
            y=average_delay_by_day.values,
            ax=ax4,
            hue=average_delay_by_day.index,  # Set hue to the same as x
            palette="viridis",
            legend=False  # Explicitly hide legend
        )
        ax4.set_xlabel("Day of Week")
        ax4.set_ylabel("Average Delay (minutes)")
        ax4.set_title("Average Train Delay by Day of the Week")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()  # Add tight_layout for this plot too
        st.pyplot(fig4)
    else:
        st.info(
            "Not enough data with valid dates and delay information to show average delay by day of the week."
        )

with col4:
    # Heatmap of correlations
    st.subheader("🔍 Correlation Heatmap")
    numeric_cols = data.select_dtypes(include=np.number)
    fig3, ax3 = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        numeric_cols.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax3,
        annot_kws={"size": 10},
    )
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    st.pyplot(fig3)

# Add more graphs below the 2x2 grid
st.markdown("---")  # Separator

# Scatter plot of Average Journey Time vs. Delay
st.subheader("Scatter Plot: Average Journey Time vs. Delay")
if (
    "Average journey time" in data.columns
    and "delay_minutes" in data.columns
    and not data.empty
):
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=data, x="Average journey time", y="delay_minutes", ax=ax5, alpha=0.6
    )
    ax5.set_xlabel("Average Journey Time (minutes)")
    ax5.set_ylabel("Delay (minutes)")
    ax5.set_title("Average Journey Time vs. Delay")
    plt.tight_layout()
    st.pyplot(fig5)
else:
    st.info("Data for Average Journey Time or Delay not available for scatter plot.")


# New row for two more plots
col5, col6 = st.columns(2)

with col5:
    # Histogram of Number of Scheduled Trains
    st.subheader("Distribution of Scheduled Trains")
    if "Number of scheduled trains" in data.columns and not data.empty:
        fig6, ax6 = plt.subplots()
        train_data = pd.DataFrame(data["Number of scheduled trains"].dropna())
        sns.histplot(data=train_data, x="Number of scheduled trains", bins=30, kde=True, ax=ax6)
        ax6.set_xlabel("Number of Scheduled Trains")
        ax6.set_ylabel("Frequency")
        ax6.set_title("Histogram of Number of Scheduled Trains")
        plt.tight_layout()
        st.pyplot(fig6)
    else:
        st.info("Data for Number of Scheduled Trains not available for histogram.")

with col6:
    # Bar plot of Average Delay by Arrival Station
    st.subheader("Average Delay by Arrival Station")
    if (
        "Arrival station" in data.columns
        and "delay_minutes" in data.columns
        and not data.empty
    ):
        # Calculate average delay by arrival station
        average_delay_by_arrival = (
            data.groupby("Arrival station")["delay_minutes"]
            .mean()
            .sort_values(ascending=False)
        )

        if not average_delay_by_arrival.empty:
            fig7, ax7 = plt.subplots(
                figsize=(15, 7)
            )  # Adjust size for potential many stations
            sns.barplot(
                x=average_delay_by_arrival.index,
                y=average_delay_by_arrival.values,
                ax=ax7,
                hue=average_delay_by_arrival.index,  # Set hue to the same as x
                palette="viridis",
                legend=False  # Explicitly hide legend
            )
            ax7.set_xlabel("Arrival Station")
            ax7.set_ylabel("Average Delay (minutes)")
            ax7.set_title("Average Train Delay by Arrival Station")
            plt.xticks(rotation=90, ha="right")  # Rotate labels
            plt.tight_layout()
            st.pyplot(fig7)
        else:
            st.info(
                "Not enough data with valid delay information to show average delay by arrival station."
            )
    else:
        st.info("Data for Arrival Station or Delay not available for bar plot.")


# Station-Level Summary
st.header("📍 Station Summary Statistics")
if not filtered_data.empty:
    stats = (
        filtered_data["delay_minutes"]
        .agg(["count", "mean", "median", "max", "min"])
        .to_frame()
        .T
    )
    st.dataframe(
        stats.rename(
            columns={
                "count": "Total Trains",
                "mean": "Average Delay (min)",
                "median": "Median Delay (min)",
                "max": "Max Delay",
                "min": "Min Delay",
            }
        )
    )
else:
    st.info("No statistics to display for this selection.")


# Prediction Interface
st.header("🧠 Predict Delay")

# Simplified features – you may need to match this with your model's input
st.markdown("Provide the required inputs for delay prediction:")

# Get the unique list of all stations
all_stations = sorted(data["Departure station"].dropna().astype(str).unique())

col1, col2 = st.columns(2)
with col1:
    # Select Departure Station from the full list
    pred_departure_station = st.selectbox(
        "Departure Station",
        all_stations,  # Use the full list
        key="pred_departure_station",
    )
    pred_hour = st.slider("Hour of Day", 0, 23, 8, key="pred_hour")

with col2:
    # Filter arrival stations to exclude the selected departure station
    available_arrival_stations = [
        s for s in all_stations if s != pred_departure_station
    ]

    # Select Arrival Station from the filtered list
    if not available_arrival_stations:
        st.warning("No other stations available for arrival.")
        pred_arrival_station = None  # No valid arrival station to select
    else:
        pred_arrival_station = st.selectbox(
            "Arrival Station",
            available_arrival_stations,  # Use the filtered list
            key="pred_arrival_station",
        )

    # Map day name to number (Monday=0, Sunday=6)
    day_name_to_num = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    pred_day_of_week_name = st.selectbox(
        "Day of Week", day_name_to_num, key="pred_day_of_week_name"
    )
    pred_day_of_week_num = day_name_to_num.index(pred_day_of_week_name)

# You must match input format with training features used in your model
# For example, you may need to one-hot encode or map categories manually
# Create a dictionary for the prediction input, including all features the model expects
# Use plausible default values for features not controlled by the user in the dashboard
input_data_dict = {
    "Departure station": pred_departure_station,
    "Arrival station": pred_arrival_station,  # Include Arrival station
    "Hour": pred_hour,
    "DayOfWeek": pred_day_of_week_num,
    "Month": date.month if date else 1,  # Handle None case with default value
    "Number of scheduled trains": (
        data["Number of scheduled trains"].mean()
        if "Number of scheduled trains" in data.columns
        else 1
    ),  # Use mean or a default
    "Number of cancelled trains": (
        data["Number of cancelled trains"].mean()
        if "Number of cancelled trains" in data.columns
        else 0
    ),  # Use mean or a default
}

# Add placeholder/mean values for percentage delay columns if they were used in training
for col in expected_features:
    if "pct delay" in col.lower() and col not in input_data_dict:
        # Try to get mean from data, fallback to 0
        input_data_dict[col] = data[col].mean() if col in data.columns else 0

# Create a DataFrame from the input dictionary
input_df_raw = pd.DataFrame([input_data_dict])

# Identify categorical columns in the raw input (should match those used for fitting OHE)
categorical_cols_raw = ["Departure station", "Arrival station"]

# Separate categorical and numerical columns in the input
input_cat = input_df_raw[categorical_cols_raw]
input_num = input_df_raw.drop(columns=categorical_cols_raw)

# Apply the loaded OneHotEncoder
input_cat_encoded = ohe.transform(input_cat).toarray()

# Create a DataFrame from the encoded categorical features with the correct column names
input_cat_encoded_df = pd.DataFrame(
    input_cat_encoded,
    columns=ohe.get_feature_names_out(categorical_cols_raw),
    index=input_df_raw.index,
)

# Concatenate the numerical and encoded categorical features
# Ensure the column order matches the expected features from training
input_df_processed = pd.concat([input_num, input_cat_encoded_df], axis=1)

# Reindex and reorder columns to match the training data (expected_features)
# This is crucial for the prediction to work correctly
try:
    input_df_final = input_df_processed.reindex(columns=expected_features, fill_value=0)
except ValueError as e:
    st.error(
        f"Error aligning prediction features: {e}. Expected features: {expected_features}, Processed features: {input_df_processed.columns.tolist()}"
    )
    input_df_final = None  # Prevent prediction if columns don't match

st.write("Model Input (Processed):")
st.dataframe(input_df_final)

if st.button("Predict Delay"):
    if input_df_final is not None:
        try:
            prediction = model.predict(input_df_final)[0]
            st.success(
                f"Predicted Delay for {pred_departure_station} to {pred_arrival_station} on {date} at {pred_hour}H: estimated delay is {prediction:.2f} minutes"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error("Prediction input could not be prepared correctly.")
