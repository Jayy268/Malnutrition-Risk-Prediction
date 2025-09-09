# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:45:27 2025

@author: Jideofor
"""

import numpy as np
import gradio as gr    
import joblib

model_path = "malnutrition_model.pkl.gz"
malnutrition_loaded_model = joblib.load(model_path)  

# Mapping for Rural Location
rural_location_map = {
    'Bungoma': 0,
    'Homa Bay': 1,
    'Kakamega': 2,
    'Kisumu': 3,
    'Migori': 4,
    'Narok': 5,
    'Siaya': 6
}

# Prediction function
def score_prediction(Weight_for_Age, Height_for_Age, Weight_for_Height, Parental_Education,
                     Household_Income_N, Dietary_Diversity, Breastfeeding_Duration,
                     Access_to_Healthcare, Clean_Water, Sanitation_Facilities,
                     Availability_of_Food, Seasonal_Variations, Market_Access, Rural_Location):

    
    rural_location_code = rural_location_map[Rural_Location]

    
    input_array = np.array([
        Weight_for_Age, Height_for_Age, Weight_for_Height, Parental_Education,
        Household_Income_N, Dietary_Diversity, Breastfeeding_Duration,
        Access_to_Healthcare, Clean_Water, Sanitation_Facilities, Availability_of_Food,
        Seasonal_Variations, Market_Access, rural_location_code
    ]).reshape(1, -1)

    # Predict
    prediction = malnutrition_loaded_model.predict(input_array)

    return f"Predicted Malnutrition Risk: {prediction[0]:,.2f}"

# Gradio Interface
inputs = [
    gr.Number(label="Weight for Age"),
    gr.Number(label="Height for Age"),
    gr.Number(label="Weight for Height"),
    gr.Number(label="Parental Education (None: 0, Primary: 1, Secondary: 2, Tertiary: 3)"),
    gr.Number(label="Household Income (Ksh)"),
    gr.Number(label="Dietary Diversity"),
    gr.Number(label="Breastfeeding Duration (months)"),
    gr.Number(label="Access to Healthcare (No: 0, Yes: 1)"),
    gr.Number(label="Access to Clean Water (No: 0, Yes: 1)"),
    gr.Number(label="Access to Sanitation Facilities (No: 0, Yes: 1)"),
    gr.Number(label="Availability of Food (No: 0, Yes: 1)"),
    gr.Number(label="Seasonal Variations (No: 0, Yes: 1)"),
    gr.Number(label="Market Access (No: 0, Yes: 1)"),
    gr.Dropdown(choices=list(rural_location_map.keys()), label="Rural Location")
]

interface = gr.Interface(
    fn=score_prediction,
    inputs=inputs,
    outputs=gr.Textbox(label="Predicted Malnutrition Risk (Low: 0, Moderate: 1, High: 2)"),
    title="Malnutrition Risk Prediction",
    description="Enter the required features to predict malnutrition risk."
)

# Launch app
if __name__ == "__main__":
    interface.launch()