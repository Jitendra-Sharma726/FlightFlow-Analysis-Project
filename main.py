import pandas as pd

FILE_NAME = "flights.csv"

# 1. Data Loading and Data Parsing
def load_flight_data(filename):
    """
    Load flight dataset and parse dates from 'FlightDate'.
    """
    try:
        # Load CSV
        df = pd.read_csv(filename)
        
        # We are checking if required column exists
        if "FlightDate" not in df.columns:
            print("Error: 'FlightDate' column is missing in the CSV.")
            return pd.DataFrame()

        # Convert 'FlightDate' column to datetime objects
        df["FlightDate"] = pd.to_datetime(df["FlightDate"], format="%Y-%m-%d")
        return df
        
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()


# 2. Data Cleaning
def clean_delay_data(df):
    """
    Assumption: If delay is NaN (empty), it means the flight was On Time (0 delay).
    """
    
    # Fill missing values (NaN) in DepartureDelay with 0
    if "DepartureDelay" in df.columns:
        df["DepartureDelay"] = df["DepartureDelay"].fillna(0)
    
    return df


# 3. Feature Extraction
def extract_day_features(df):
    """
    Extract the 'Day Name' (Monday, Tuesday...) from the Date column.
    """
    
    # Extract day name using the .dt accessor
    df["Day_Name"] = df["FlightDate"].dt.day_name()
    
    return df


# 4. Statistical Analysis
def airline_reliability_stats(df):
    """
    Calculate the Average Departure Delay and Total Flights for each airline.
    """
    stats = df.groupby("Airline")["DepartureDelay"].agg([
        "mean", 
        "count"
    ])
    
    # Rename columns for clarity
    stats = stats.rename(columns={"mean": "Avg_Delay", "count": "Total_Flights"})
    
    # Round the average delay to 2 decimal places
    stats["Avg_Delay"] = stats["Avg_Delay"].round(2)
    
    return stats


# 5. Ranking & Sorting
def rank_airlines(stats_df):
    """
    Rank airlines from Most Reliable (Lowest Delay) to Least Reliable.
    """
    return stats_df.sort_values("Avg_Delay", ascending=True)


# 6. Pattern Identification
def identify_delayed_days(df):
    """
    Find which day of the week has the worst delays on average.
    """
    day_stats = df.groupby("Day_Name")["DepartureDelay"].mean()
    
    # IMPROVEMENT: Round this result too for cleaner output
    return day_stats.sort_values(ascending=False).round(2)


if __name__ == "__main__":
    print("### FlightFlow Analysis ###")

    # 1. Load Data
    raw_df = load_flight_data(FILE_NAME)

    if not raw_df.empty:
        # 2. Clean Data
        clean_df = clean_delay_data(raw_df)
        print(f"Data Loaded & Cleaned: {clean_df.shape[0]} flights processed.")

        # 3. Feature Extraction
        enhanced_df = extract_day_features(clean_df)
        
        # 4. Reliability Analysis
        stats = airline_reliability_stats(enhanced_df)
        ranked = rank_airlines(stats)
        
        print("\nMost Reliable Airlines (Lowest Avg Delay):")
        print(ranked.head(3))
        
        print("\nLeast Reliable Airlines (Highest Avg Delay):")
        print(ranked.tail(3))

        # 5. Pattern Analysis
        worst_days = identify_delayed_days(enhanced_df)

        print("\nWorst Days to Fly (Highest Avg Delay):")
        print(worst_days.head(3))
    
    else:
        print("Analysis could not proceed due to data loading errors.")


