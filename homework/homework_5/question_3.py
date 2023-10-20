import pickle

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

client = {
    "job": "retired",
    "duration": 445,
    "poutcome": "success"
    }

X = dv.transform([client])
score = model.predict_proba(X)[0, 1]

print(score)