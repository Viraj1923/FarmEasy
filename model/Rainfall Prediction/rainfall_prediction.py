import pandas as pd
import sys

# Load the dataset into a dataframe
df = pd.read_csv(r'Farmmm\Farm\model\Rainfall Prediction\rainfall_in_india_1901-2015.csv')


# Define a function to predict rainfall for a given district and month
def predict_rainfall(state, month):
    # Filter the dataframe to only include rows with the given district
    state_data = df[df['SUBDIVISION'] == state]

    # Calculate the average rainfall for the given month across all the years
    avg_rainfall = state_data[month].mean()
    
    # Return the predicted rainfall for the given month
    return avg_rainfall




if len(sys.argv) < 2:
    print("Please provide the region as a command-line argument.")
    sys.exit(1)  # Exit the script with an error code

Jregion = sys.argv[1]
# Get the input parameters as command line arguments

Jmonth = sys.argv[2]


#predicted_rainfall = predict_rainfall('ANDAMAN & NICOBAR ISLANDS', 'JAN')

predicted_rainfall = predict_rainfall(Jregion, Jmonth)
print(predicted_rainfall)


