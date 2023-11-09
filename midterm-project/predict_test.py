import requests

url = 'http://localhost:7777/predict'

patient = {
    "gender": "male",
    "age": 65.0,
    "hypertension": 1,
    "heart_disease": 0,
    "ever_married": 1,
    "work_type": "private",
    "residence_type": "urban",
    "avg_glucose_level": 87.7,
    "bmi": 22.5,
    "smoking_status": "never_smoked"
}

response = requests.post(url, json=patient).json()

print(response)
print()

if response['stroke'] == True:
    print('The patient needs a doctor and urgent medical treatment! Big risk of CVA!')
else:
    print('There are no big risk of stroke for the patient.')