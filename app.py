# install libraries ---
# pip install fastapi uvicorn 

# 1. Library imports
import uvicorn
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
import pickle

# 2. Create the app object
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. load the model
rgModel = pickle.load(open("model.pkl", "rb"))

# 4. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get("/predictLoan_Status")
def gePredictLoan_Status(Gender: int, Married: int, Dependents:int, Education:int, Self_Employed:int, ApplicantIncome:int, CoapplicantIncome:float, LoanAmount:float, Loan_Amount_Term:float, Credit_History:float, Property_Area:int):
    prediction = rgModel.predict([[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]])
    return {'Loan_Status': str(prediction[0])}

# 5. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, port=80, host='0.0.0.0')
    
# uvicorn app:app --host 0.0.0.0 --port 80
# http://127.0.0.1/predictLoan_Status?Gender=1&Married=1&Dependents=1&Education=1&Self_Employed=0&ApplicantIncome=4583&CoapplicantIncome=1508.0&LoanAmount=128.0&Loan_Amount_Term=360.0&Credit_History=1.0&Property_Area=0