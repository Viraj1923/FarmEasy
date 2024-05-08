import os
import subprocess
from flask import Flask, render_template, request, send_from_directory
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings

#--------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------
# Filter out any warnings you want to suppress
warnings.filterwarnings("ignore", category=UserWarning)

# Load the dataset into a dataframe
df = pd.read_csv(r'model\Rainfall Prediction\rainfall_in_india_1901-2015.csv')
def predict_rainfall(state, month):
    state = state.lower()
    state_data = df[df['SUBDIVISION'].str.lower() == state]
    
    # Check if data is available for the selected state
    if state_data.empty:
        return "Data not available for the selected state."
    
    avg_rainfall = state_data[month].mean()
    return avg_rainfall


data = pd.read_csv(r"model\Fertilizer Recommendation\fertilizer_recommendation_csv.csv")
# Label encoding for categorical features
le_soil = LabelEncoder()
data['Soil Type'] = le_soil.fit_transform(data['Soil Type'])
le_crop = LabelEncoder()
data['Crop Type'] = le_crop.fit_transform(data['Crop Type'])

# Splitting the data into input and output variables
X = data.iloc[:, :8]
y = data.iloc[:, -1]

# Training the Decision Tree Classifier model
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X, y)


#--------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------


app = Flask(__name__, template_folder='templates')


#crop_prediction_model = joblib.load(r'Farmmm\Farm\model\Crop Prediction\crop_pred.joblib')
model= joblib.load(r'model\Crop Recommendation\crop_app')

#--------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------



@app.route('/')
def home():
    return render_template('demo.html', title='FarmEasy-Home')

@app.route('/crop_pr')
def crop_pr():
    return render_template('Crop Prediction.html', title='FarmEasy-Crop Prediction')

@app.route('/crop_recommendation')
def crop_recommendation():
    return render_template('Crop Recommendation.html', title='FarmEasy-Crop Recommendation')

@app.route('/ra_pre')
def ra_pre():
    return render_template('Rainfall Prediction.html', title='FarmEasy-Rainfall Prediction')

@app.route('/yield_pr')
def yield_pr():
    return render_template('Yield Prediction.html', title='FarmEasy-Yield Prediction')

@app.route('/fert_recommendation')
def fer_recommendation():
    return render_template('Fertilizer Recommendation.html', title='FarmEasy-Fertilizer Recommendation')


# Load static files
@app.route('/static/<path:filename>')
def send_static(filename):
    return send_from_directory('static', filename)



#--------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------


@app.route('/predict_rain', methods=['POST'])
def predict_rain():
    if request.method == 'POST':
        region = request.form['region']
        month = request.form['month']
        
        # Call the predict_rainfall function
        output = predict_rainfall(region.lower(), month)

        return render_template('Rainfall Prediction.html', prediction=output)
    else:
        return "Invalid request method."


@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    if request.method == 'POST':
        # Get form data
        state = request.form['state']
        district = request.form['district']
        season = request.form['season']
        crop = request.form['crop']
        area = request.form['area']
        
        # Call the script with user input as arguments
        cmd = ['python', 'model\Yield Prediction\yield_prediction.py', state, district, season, crop, area]
        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        prediction = result.stdout.decode('utf-8').strip()

        # Render template with prediction result
        return render_template('Yield Prediction.html', prediction=prediction)






'''@app.route('/predict_crop', methods=['GET', 'POST'])
def crop_prediction():
    if request.method == 'POST':
        state = request.form['State_Name']
        district = request.form['District_Name']
        season = request.form['Season']

        prediction = crop_prediction_model.predict([[state, district, season]])  
        
        return render_template('Crop Prediction.html', prediction=prediction)
    else:
        return render_template('Crop Prediction.html')'''


@app.route('/rec_cr', methods=["POST"])
def brain():
    Nitrogen=float(request.form['Nitrogen'])
    Phosphorus=float(request.form['Phosporus'])
    Potassium=float(request.form['Potassium'])
    Temperature=float(request.form['Temperature'])
    Humidity=float(request.form['Humidity'])
    Ph=float(request.form['ph'])
    Rainfall=float(request.form['Rainfall'])
     
    values=[Nitrogen,Phosphorus,Potassium,Temperature,Humidity,Ph,Rainfall]
    
    if Ph>0 and Ph<=14 and Temperature<100 and Humidity>0:
        
        arr = [values]
        acc = model.predict(arr)
        print(acc)
        return render_template('Crop Recommendation.html', result=str(acc))
    else:
        return "Sorry...  Error in entered values in the form Please check the values and fill it again"


@app.route('/recommend_fertilizer', methods=['POST'])
def fertilizer_recommendation():
    if request.method == 'POST':
        # Receive input parameters from the form
        temp = request.form['Temperature']
        humidity = request.form['Humidity']
        soil_moisture = request.form['Soil Moisture']
        soil = request.form['Soil Type']
        crop = request.form['Crop Type']
        nitrogen = request.form['Nitrogen']
        potassium = request.form['Potassium']
        phosphorus = request.form['Phosphorus']

        # Encode categorical variables
        soil_enc = le_soil.transform([soil])[0]
        crop_enc = le_crop.transform([crop])[0]

        # Call the fertilizer recommendation script with input parameters
        user_input = [[temp, humidity, soil_moisture, nitrogen, phosphorus, potassium, soil_enc, crop_enc]]
        fertilizer_name = dtc.predict(user_input)

        # Render template with the prediction result
        return render_template('Fertilizer Recommendation.html', result=fertilizer_name[0])


#--------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    app.run(debug=True)
