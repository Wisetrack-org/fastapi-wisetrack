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

# DB_HOST = "127.0.0.1"  # From Railway or wherever your DB is hosted
# DB_PORT = 3306 # From Railway or wherever your DB is hosted
# DB_USER = "root"
# DB_PASSWORD = "root"
# DB_NAME = "wisetrack"

# Function to connect to the database
def get_db_connection():
    try:
        mydb = mysql.connector.connect(
            host=os.environ.get("HOST"),
            port=os.environ.get("PORT", 3306),
            user="root",
            password=os.environ.get("PASSWORD"),
            database=os.environ.get("DATABASE")
        )
        return mydb
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        raise  # Re-raise the exception to be handled by the caller


# Load the trained model
model_path = os.path.join('models', 'final_model.pkl')
model_combined = joblib.load(model_path)

# Define the schema for input data (using your provided attributes)
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
    try:
        # Convert the Pydantic model to a dictionary
        data_dict = data.dict()
        
        # Create a new dictionary with only the columns the model expects
        model_features = {}
        
        # Copy most fields directly
        for field in data_dict:
            model_features[field] = data_dict[field]

        # Calculate Engagement_Score if needed (using original values)
        # Note: If Engagement_Score is already provided, you might want to use that instead
        # model_features['Engagement_Score'] = 0.6 * data_dict['Participation'] + 0.4 * data_dict['Social_Engagement']
        
        # Create DataFrame from the transformed dictionary
        df = pd.DataFrame([model_features])
        
        # Print the columns to debug
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame shape: {df.shape}")
        
        print(f"Input feature values: {df.iloc[0].to_dict()}") 
        # Make prediction
        prediction = model_combined.predict(df)
        
        print(f"Raw prediction type: {type(prediction)}")
        print(f"Raw prediction value: {prediction}")
        if hasattr(prediction, 'shape'):
            print(f"Prediction shape: {prediction.shape}")
        # Handle prediction output
        if hasattr(prediction, 'ndim') and prediction.ndim > 0:
            pred_value = prediction[0]  # Get the first element if it's an array
        else:
            pred_value = prediction  # Otherwise use as is
            
        print(f"Prediction value: {pred_value}")
            
        # Interpret the prediction result
        risk_status_str = "Low Risk"
        # risk_type = "No Risk"
            
        if hasattr(model_combined, 'predict_proba'):
            probas = model_combined.predict_proba(df)
            print(f"Prediction probabilities: {probas}")

        low_thresh = 0.5
        med_thresh = 0.7
        
        max_proba = probas[0].max()
        
        if max_proba > med_thresh:
            risk_status_str = "No Risk"
        elif max_proba < med_thresh and max_proba > low_thresh:
            risk_status_str = "Risk"
        elif max_proba < low_thresh:
            risk_status_str = "High Risk"
        else:
            risk_status_str = "Low Risk"

        try:
            mydb = get_db_connection()
            mycursor = mydb.cursor()

            # Assuming you have a way to identify the student (e.g., student_id)

            sql = "UPDATE Students SET at_risk = %s WHERE student_id = %s"
            val = (risk_status_str, 1)  # Use the calculated risk status
            mycursor.execute(sql, val)
            mydb.commit()

            print(f"{mycursor.rowcount} record(s) updated")

        except mysql.connector.Error as err:
            print(f"Database update error: {err}")
            raise HTTPException(status_code=500, detail=f"Database update error: {err}") # Return error to the caller

        finally:
            if mydb.is_connected():
                mycursor.close()
                mydb.close()
        
        return {
            "Risk_Status": risk_status_str,
            "Debug_Info": {
                "Feature_Count": df.shape[1],
                "Features": df.columns.tolist()
            }
        }
            
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}