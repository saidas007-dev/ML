import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load the model
model = pickle.load(open('diabetes_prediction_model.pkl', 'rb'))

def categorize_bmi(bmi):
    if pd.isna(bmi) or bmi == 0: return np.nan
    if bmi < 18.5: return 0
    elif bmi < 25: return 1
    elif bmi < 30: return 2
    else: return 3

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, bmi_cat_manual):
    # Clamping based on exact dataset min/max values
    pregnancies = clamp(pregnancies, 0, 17)
    glucose = clamp(glucose, 0, 199)
    blood_pressure = clamp(blood_pressure, 0, 130.0)
    skin_thickness = clamp(skin_thickness, 0, 100.0)
    insulin = clamp(insulin, 0, 850)
    bmi = clamp(bmi, 0.0, 70.0)
    dpf = clamp(dpf, 0.078, 2.42)
    age = clamp(age, 20, 85)

    if bmi_cat_manual == "Auto":
        bmi_category = categorize_bmi(bmi)
    else:
        bmi_category = int(bmi_cat_manual)

    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age, bmi_category
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'BMI_Category'
    ])

    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    input_df[cols_with_zeros] = input_df[cols_with_zeros].replace(0, np.nan)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"
    return f"{result} (Probability: {probability:.2%})"


interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        # Dropdown goes up to 17 (dataset max)
        gr.Dropdown(choices=list(range(0, 18)), value=0, label="Pregnancies"),

        # Sliders mapped to dataset min and max. Values default near the dataset median.
        gr.Slider(0, 199, value=117, step=1, label="Glucose (mg/dL)"),
        gr.Slider(0, 130, value=72, step=1, label="Blood Pressure (mm Hg)"),
        gr.Slider(0, 100.0, value=23, step=1, label="Skin Thickness (mm)"),
        gr.Slider(0, 846, value=30, step=1, label="Insulin (mu U/ml)"),

        gr.Slider(0.0, 70.0, value=32.0, step=0.1, label="BMI"),

        gr.Number(label="Diabetes Pedigree Function", minimum=0.078, maximum=2.42, value=0.372),

        gr.Slider(20, 85, value=35, step=1, label="Age (years)"),

        gr.Radio(
            choices=["Auto", "0", "1", "2", "3"],
            value="Auto",
            label="BMI Category (0=Underweight, 1=Normal, 2=Overweight, 3=Obese)"
        )
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Diabetes Prediction Machine Learning App",
    description="Interactive input controls constrained by the real maximum and minimum limits found in the Pima Indians dataset."
)

if __name__ == "__main__":
    interface.launch()
