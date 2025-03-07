from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
import mysql.connector
from dotenv import load_dotenv  # For local development

# Load environment variables from .env file (for local development)
load_dotenv()  

app = FastAPI()

origins = [
    "https://wisetrack.vercel.app",  # Your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to connect to the database
def get_db_connection():
    try:
        mydb = mysql.connector.connect(
            # host=os.environ.get("HOST"),
            host="127.0.0.1",
            port=os.environ.get("PORT", 3306),
            user="root",
            # password=os.environ.get("PASSWORD"),
            password="root",
            # database=os.environ.get("DATABASE")
            database="wisetrack"
        )
        return mydb
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        raise  # Re-raise the exception to be handled by the caller


# Load the trained model
model_path = os.path.join('models', 'final_model.pkl')
model_combined = joblib.load(model_path)

class StudentData(BaseModel):
    Backlogs: int
    Scholarship: int
    Tuition_Delay: int
    Financial_Stress_Index: float
    Finance_Issue: int
    Work_Hours: int
    Academic_Consistency: float
    Homework_Streak: int
    Grades: int
    Weekly_Test_Scores: int
    Attention_Score: int
    Attendance: int
    Absences: int
    Engagement_Score: float
    Not_Interested: int
    Discrimination: int
    Participation: int
    Physical_Health_Issues: int
    Social_Engagement: int
    Mental_Health_Issues: int
    Physical_Disability: int
    School_Far: int
    Sudden_Drop: int

@app.post("/predict")
def predict_risk(data: StudentData):
    # mydb = None
    try:
        data_dict = data.dict()
        
        model_features = {}
        
        for field in data_dict:
            model_features[field] = data_dict[field]

        df = pd.DataFrame([model_features])
        
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame shape: {df.shape}")
        
        print(f"Input feature values: {df.iloc[0].to_dict()}") 
        prediction = model_combined.predict(df)
        
        print(f"Raw prediction type: {type(prediction)}")
        print(f"Raw prediction value: {prediction}")
        if hasattr(prediction, 'shape'):
            print(f"Prediction shape: {prediction.shape}")

        if hasattr(prediction, 'ndim') and prediction.ndim > 0:
            pred_value = prediction[0]
        else:
            pred_value = prediction
            
        print(f"Prediction value: {pred_value}")
            
        risk_status_str = "Low Risk"
            
        if hasattr(model_combined, 'predict_proba'):
            probas = model_combined.predict_proba(df)
            print(f"Prediction probabilities: {probas}")

        low_thresh = 0.5
        med_thresh = 0.7
        
        max_proba = probas[0].max()
        
        if max_proba > med_thresh:
            risk_status_str = "No Risk"
        elif max_proba < med_thresh and max_proba > low_thresh:
            risk_status_str = "High Risk"
        elif max_proba < low_thresh:
            risk_status_str = "High Risk"
        else:
            risk_status_str = "Low Risk"

        try:
            mydb = get_db_connection()
            mycursor = mydb.cursor()


            sql = "UPDATE Students SET at_risk = %s WHERE student_id = %s"
            val = (risk_status_str, 1)
            mycursor.execute(sql, val)
            mydb.commit()

            print(f"{mycursor.rowcount} record(s) updated")

        except mysql.connector.Error as err:
            print(f"Database update error: {err}")
            raise HTTPException(status_code=500, detail=f"Database update error: {err}")

        finally:
            if mydb.is_connected():
                mycursor.close()
                mydb.close()
        
        return {
            "Risk_Status": risk_status_str,
        }
            
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}