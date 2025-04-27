import gradio as gr
import joblib
import numpy as np
from xgboost import XGBClassifier

# Load your trained model
with open('xgboost-model.pkl', 'rb') as file:
    model = joblib.load(file)

# Function for prediction
def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes,
                     ejection_fraction, high_blood_pressure, platelets,
                     serum_creatinine, serum_sodium, sex, smoking, time):
    features = [[age, anaemia, creatinine_phosphokinase, diabetes,
                 ejection_fraction, high_blood_pressure, platelets,
                 serum_creatinine, serum_sodium, sex, smoking, time]]
    prediction = model.predict(features)
    return "Survived" if prediction[0] == 0 else "Death"

age_slider = gr.Slider(minimum=40, maximum=95, value=60, label="Age")
creatinine_phosphokinase_slider = gr.Slider(minimum=23, maximum=7861, label="Creatinine Phosphokinase")
ejection_fraction_slider = gr.Slider(minimum=14, maximum=80, value=30, label="Ejection Fraction")
platelets_slider = gr.Slider(minimum=25100, maximum=850000, value=263358, label="Platelets")
serum_creatinine_slider = gr.Slider(minimum=0.5, maximum=9.4, value=1.1, label="Serum Creatinine")
serum_sodium_slider = gr.Slider(minimum=113, maximum=148, value=137, label="Serum Sodium")
time_slider = gr.Slider(minimum=4, maximum=285, value=130, label="Time")

anaemia_radio = gr.Radio(choices=[0, 1], type="index", label="Anaemia")
diabetes_radio = gr.Radio(choices=[0, 1], type="index", label="Diabetes")
high_blood_pressure_radio = gr.Radio(choices=[0, 1], type="index", label="High Blood Pressure")
sex_radio = gr.Radio(choices=[0, 1], type="index", label="Sex (0: Female, 1: Male)")
smoking_radio = gr.Radio(choices=[0, 1], type="index", label="Smoking")

# Inputs from user
inputs = [age_slider, anaemia_radio, creatinine_phosphokinase_slider, diabetes_radio,
          ejection_fraction_slider, high_blood_pressure_radio, platelets_slider,
          serum_creatinine_slider, serum_sodium_slider, sex_radio, smoking_radio, time_slider]

# Output response
output = gr.Textbox(label="Prediction")

if __name__ == "__main__":
    # Gradio interface to generate UI link
    title = "Patient Survival Prediction"
    description = "Predict survival of patient with heart failure, given their clinical record"

    iface = gr.Interface(fn = predict_death_event,
                            inputs = inputs,
                            outputs = output,
                            title = title,
                            description = description,
                            flagging_mode='never')

    iface.launch(share = True, server_name="0.0.0.0", server_port = 8082)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface
