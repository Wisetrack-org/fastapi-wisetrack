from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()


# Load the trained model
model_path = os.path.join('models', 'xgboost_multi_output_pipeline_model.pkl')

model_combined = joblib.load(model_path)


# Define the schema for input data
class StudentData(BaseModel):
    Attendance: float
    Grades: float
    Homework_Streak: int  # Adjusted schema
    Feedback_Behavior: int
    Weekly_Test_Scores: float  # Adjusted schema
    Attention_Test_Scores: float  # Adjusted schema
    Ragging: int
    Finance_Issue: int
    Mental_Health_Issue: int
    Physical_Health_Issue: int
    Discrimination_Based_on_Gender: int
    Physical_Disability: int
    Not_Interested: int
    Working_and_Studying: int
    School_Is_Far: int

@app.post("/predict")
def predict_risk(data: StudentData):
    try:
        # Convert input data into a DataFrame
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])

        df.rename(columns={
            'Attendance': 'Attendance',
            'Grades': 'Grades',
            'Homework_Streak': 'Homework Streak',
            'Feedback_Behavior': 'Feedback Behavior',
            'Weekly_Test_Scores': 'Weekly Test Scores',
            'Attention_Test_Scores': 'Attention Test Scores',
            'Ragging': 'Ragging',
            'Finance_Issue': 'Finance Issue',
            'Mental_Health_Issue': 'Mental Health Issue',
            'Physical_Health_Issue': 'Physical Health Issue',
            'Discrimination_Based_on_Gender': 'Discrimination Based on Gender',
            'Physical_Disability': 'Physical Disability',
            'Not_Interested': 'Not Interested',
            'Working_and_Studying': 'Work and Learn',
            'School_Is_Far': 'School is Far Off'
        }, inplace=True)
        
        
        # prediction = model_combined.predict(df)[0]
        prediction = model_combined.predict(df)
        # Check the shape of the prediction
        if prediction.ndim == 1 and prediction.size == 1:
            pred_value = prediction[5]  # Get the single value
        elif prediction.ndim == 2 and prediction.shape[0] == 1:
            pred_value = prediction[0, 1]  # Get the first value from a 2D array
        else:
            return {"error": "Unexpected prediction shape."}

        # Interpret the prediction result
        risk_status_str = "Low Risk"
        risk_type = "No Risk"

        if pred_value == 0:
            risk_status_str = "Low Risk"
            risk_type = "No Risk"
        else:
            risk_status_str = "High Risk"
            risk_types = ["Academic Risk", "Financial Risk", "Mental Health Risk", "Bullying Risk"]
            risk_type_index = pred_value - 1
            
            if risk_type_index < len(risk_types):
                risk_type = risk_types[risk_type_index]
            else:
                risk_type = "Unknown Risk Type"

        return {
            "Risk_Status": risk_status_str,
            "Risk_Type": risk_type
        }

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
    
    
    
    
    
    
    #     print(prediction)
    #     print(type(prediction))  # This will show you whether it's <class 'int'> or <class 'float'>


    #     if prediction == 0:
    #         risk_status_str = "Low Risk"
    #         risk_type = "No Risk"
    #     else:
    #         risk_status_str = "High Risk"
    #         # Assuming the model outputs a class index for risk types
    #         risk_types = ["Academic Risk", "Financial Risk", "Mental Health Risk", "Bullying Risk"]
    #         risk_type = risk_types[prediction - 1] if prediction - 1 < len(risk_types) else "Unknown Risk Type"

    #     return {
    #         "Risk_Status": risk_status_str,
    #         "Risk_Type": risk_type
    #     }

    # except KeyError as e:
    #     return {"error": f"KeyError during prediction: {e}"}
    # except Exception as e:
    #     return {"error": f"An error occurred: {str(e)}"}        