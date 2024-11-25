import pandas as pd
import numpy as np

# Function to generate a synthetic sales dataset
def generate_sales_data(output_file='sales_data.csv', num_months=24):
    """
    Generate a synthetic sales dataset and save it to a CSV file.

    Parameters:
        output_file (str): The name of the output CSV file.
        num_months (int): The number of months to generate data for.
    """
    # Generate monthly date range
    dates = pd.date_range(start="2023-01-01", periods=num_months, freq="M")

    # Generate synthetic sales data
    sales = [200 + np.random.randint(-20, 20) + i * 10 for i in range(num_months)]

    # Create a DataFrame
    data = pd.DataFrame({
        'Month': dates,
        'Sales': sales
    })

    # Save to CSV
    data.to_csv(output_file, index=False)
    print(f"Dataset generated and saved to {output_file}")

# Generate the dataset
generate_sales_data()
