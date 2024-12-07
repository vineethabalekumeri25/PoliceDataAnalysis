import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Preprocessing function (from preprocess.py)
def preprocess_data():
    # Reading the police dataset and other datasets with appropriate encoding
    police_data = pd.read_csv("../data/Deaths_by_Police_US.csv", encoding='ISO-8859-1')  # Adjust the path as necessary
    income_data = pd.read_csv("../data/Median_Household_Income_2015.csv", encoding='ISO-8859-1')  # Adjust the path as necessary
    hs_data = pd.read_csv("../data/Pct_Over_25_Completed_High_School.csv", encoding='ISO-8859-1')  # Adjust the path as necessary
    poverty_data = pd.read_csv("../data/Pct_People_Below_Poverty_Level.csv", encoding='ISO-8859-1')  # Adjust the path as necessary
    race_data = pd.read_csv("../data/Share_of_Race_By_City.csv", encoding='ISO-8859-1')  # Adjust the path as necessary

    # Convert all column names to lowercase for consistency
    police_data.columns = police_data.columns.str.lower()
    income_data.columns = income_data.columns.str.lower()
    hs_data.columns = hs_data.columns.str.lower()
    poverty_data.columns = poverty_data.columns.str.lower()
    race_data.columns = race_data.columns.str.lower()

    # Clean the 'city' column in all datasets
    # Remove " city", " town" or any other unwanted suffixes from city names
    police_data['city'] = police_data['city'].str.replace(r'\s*(city|town)$', '', regex=True).str.strip()
    income_data['city'] = income_data['city'].str.replace(r'\s*(city|town)$', '', regex=True).str.strip()
    hs_data['city'] = hs_data['city'].str.replace(r'\s*(city|town)$', '', regex=True).str.strip()
    poverty_data['city'] = poverty_data['city'].str.replace(r'\s*(city|town)$', '', regex=True).str.strip()
    race_data['city'] = race_data['city'].str.replace(r'\s*(city|town)$', '', regex=True).str.strip()

    # Rename columns for clarity in merged data
    income_data = income_data.rename(columns={'geographic area': 'geographic_area_income'})
    hs_data = hs_data.rename(columns={'geographic area': 'geographic_area_hs'})
    poverty_data = poverty_data.rename(columns={'geographic area': 'geographic_area_poverty'})
    race_data = race_data.rename(columns={'geographic area': 'geographic_area_race'})

    # Drop duplicates based on 'city'
    police_data = police_data.drop_duplicates(subset=['city'])
    income_data = income_data.drop_duplicates(subset=['city'])
    hs_data = hs_data.drop_duplicates(subset=['city'])
    poverty_data = poverty_data.drop_duplicates(subset=['city'])
    race_data = race_data.drop_duplicates(subset=['city'])

    def clean_and_deduplicate(data, key):
        data[key] = data[key].str.replace(r'\s*(city|town)$', '', regex=True).str.strip().str.lower()
        return data.drop_duplicates(subset=[key])

    # Apply cleaning
    police_data = clean_and_deduplicate(police_data, 'city')
    income_data = clean_and_deduplicate(income_data, 'city')
    hs_data = clean_and_deduplicate(hs_data, 'city')
    poverty_data = clean_and_deduplicate(poverty_data, 'city')
    race_data = clean_and_deduplicate(race_data, 'city')

    # Merge datasets
    merged_data = pd.merge(police_data, income_data, on="city", how="left")
    merged_data = pd.merge(merged_data, hs_data, on="city", how="left")
    merged_data = pd.merge(merged_data, poverty_data, on="city", how="left")
    merged_data = pd.merge(merged_data, race_data, on="city", how="left")

    print(f"Rows after merging with income_data: {merged_data.shape[0]}")
    print(f"Rows after merging with hs_data: {merged_data.shape[0]}")
    print(f"Rows after merging with poverty_data: {merged_data.shape[0]}")
    print(f"Rows after merging with race_data: {merged_data.shape[0]}")

    # Verify rows
    print(f"Final row count: {merged_data.shape[0]}")

    # Check for duplicates in the final dataset
    duplicates = merged_data[merged_data.duplicated(subset=['city'], keep=False)]
    print("Duplicates in merged data (if any):")
    print(duplicates)

    # Preview data
    print("Merged Data Preview:")
    print(merged_data)

    merged_data.columns = merged_data.columns.str.strip()
    print(merged_data.columns)  # Check columns again after stripping spaces and check if 'share_asian' exists in merged data

    # Rename columns for clarity after merge
    merged_data.rename(columns={
        "median income": "median_income",
        "percent_completed_hs": "high_school_completion",
        "poverty_rate": "poverty_rate",
        "share_white": "white_share",
        "share_black": "black_share",
        "share_native_american": "native_american_share",
        "share_asian": "asian_share",
        "share_hispanic": "hispanic_share"
    }, inplace=True)

    # Convert the columns related to share and other numeric columns to numeric
    merged_data['asian_share'] = pd.to_numeric(merged_data['asian_share'].str.replace(',', '').str.strip(), errors='coerce')
    merged_data['hispanic_share'] = pd.to_numeric(merged_data['hispanic_share'].str.replace(',', '').str.strip(), errors='coerce')

    # Ensure 'manner_of_death' remains as categorical (no encoding)
    merged_data['manner_of_death'] = merged_data['manner_of_death'].fillna('Unknown')

    return merged_data


# Analysis function (from analysis.py)
def analyze_data():
    # Get the preprocessed and merged data
    data = preprocess_data()

    # Print the selected columns
    print(data[['id', 'name', 'city', 'manner_of_death', 'median_income', 'poverty_rate',
                'high_school_completion', 'white_share', 'black_share',
                'native_american_share', 'asian_share', 'hispanic_share']])

    # Clean the data: Convert columns to numeric, coercing errors to NaN
    numeric_columns = ['median_income', 'poverty_rate', 'high_school_completion',
                       'white_share', 'black_share', 'native_american_share',
                       'asian_share', 'hispanic_share']
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Handle categorical column "manner_of_death" by encoding
    if 'manner_of_death' in data.columns:
        # Convert the 'manner_of_death' to a numeric code (Label Encoding)
        le = LabelEncoder()
        data['manner_of_death_encoded'] = le.fit_transform(data['manner_of_death'].astype(str))

    # Perform the correlation calculation on only the numeric columns (exclude 'manner_of_death' for now)
    correlation_columns = numeric_columns + ['manner_of_death_encoded']  # Add the encoded 'manner_of_death' for correlation
    correlation = data[correlation_columns].corr()
    print("Correlation Analysis or Correlation Matrix:")
    print(correlation)

    # Example 2: Relationship between fatalities and median income
    income_fatalities = data[['manner_of_death_encoded', 'median_income']]
    print("\nRelationship between Fatalities and Median Income:")
    print(income_fatalities.describe())

    # Example 3: Grouping by city and summarizing data
    city_summary = data.groupby('city').agg({
        'manner_of_death_encoded': 'sum',  # Total fatalities per city
        'median_income': 'mean',  # Average median income per city
        'poverty_rate': 'mean',  # Average poverty rate per city
        'high_school_completion': 'mean',  # Average graduation rate per city
        'white_share': 'mean',
        'black_share': 'mean',
        'native_american_share': 'mean',
        'asian_share': 'mean',
        'hispanic_share': 'mean'
    }).reset_index()

    print("\nCitywise Summary:")
    print(city_summary)
    return correlation, income_fatalities, city_summary.head(100)


# Run the analysis
analyze_data()
