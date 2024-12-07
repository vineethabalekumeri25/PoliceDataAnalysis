import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from analysis import analyze_data  # Import the analysis results

rcParams['font.family'] = 'DejaVu Sans'

# Plotting correlation matrix
def plot_correlation(correlation, save_path="plots/correlation_matrix.png"):
    if correlation is None or correlation.empty:
        print("Error: Correlation matrix is empty or None.")
        return
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Analysis or Correlation Matrix of Key Metrics')
    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=10
    )
    plt.yticks(rotation=0, fontsize=10)
    plt.savefig(save_path)  # Save the plot
    plt.show()
    print(f"Saved correlation plot to {save_path}")


# Plotting median income vs fatalities
def plot_income_vs_fatalities(income_fatalities, save_path="plots/income_vs_fatalities.png"):
    plt.figure(figsize=(10, 10))
    sns.boxplot(x='manner_of_death_encoded', y='median_income', data=income_fatalities)
    plt.title('Relationship Between Median Income and Fatalities')
    plt.xticks(rotation=90, fontsize=8)  # Rotate and resize labels for better readability
    plt.yticks(rotation=90, fontsize=8)
    plt.xlabel('Fatalities (Encoded)')
    plt.ylabel('Median Income')
    plt.savefig(save_path)  # Save the plot
    plt.show()
    plt.close()
    print(f"Saved income vs fatalities plot to {save_path}")

# Bar chart for city summary (e.g., median income by city)
def plot_city_summary(city_summary):
    # Replace unsupported characters in city names if necessary
    top_cities = city_summary.nlargest(10, 'median_income')
    city_summary['city'] = city_summary['city'].str.replace('\x96', '-')

    plt.figure(figsize=(10, 10))  # Larger figure size to accommodate more bars
    sns.barplot(x='city', y='median_income', data=city_summary)
    plt.xticks(rotation=90, fontsize=8)  # Rotate and resize labels for better readability
    plt.yticks(rotation=90, fontsize=8)
    plt.title('Top 100 Cities by Median Income')
    plt.xlabel('City')
    plt.ylabel('Median Income')
    plt.show()
    plt.close()
    # Save the plot with a font that supports Unicode
    save_path = "plots/citywise_median_income.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved city summary plot to {save_path}")


if __name__ == "__main__":
    correlation, income_fatalities, city_summary = analyze_data()

    # Ensure the plots folder exists
    import os
    os.makedirs("plots", exist_ok=True)

    # Generate and save the plots
    plot_correlation(correlation)
    plot_income_vs_fatalities(income_fatalities)
    plot_city_summary(city_summary)