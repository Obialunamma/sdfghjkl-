# import necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import pickle

# load data
d = pd.read_csv('data/Telco-Customer-Churn.csv')

# Preprocessing
d.drop(["customerID"], axis = 1, inplace = True) # irrelevant column

d['TotalCharges'] = pd.to_numeric(d['TotalCharges'], errors='coerce')


encoders = {}

# Encode categorical columns and save the encoders
for col in d.columns:
    if d[col].dtype == "object":
        encoders[col] = LabelEncoder()
        d[col] = encoders[col].fit_transform(d[col])
        
# Separate features and labels
y = d.Churn
x = d.drop(["Churn"], axis = 1)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# scaling 
sc = StandardScaler()
x_train_scale  = sc.fit_transform(x_train)
x_test_scale  = sc.transform(x_test)

# Initialize and train the model
svc = SVC(kernel ='poly', degree =5)
svc.fit(x_train_scale, y_train)
svc.score(x_train_scale, y_train)

cv = cross_val_score(svc, x_train_scale, y_train, cv=5, scoring='accuracy')
print(cv)
print(cv.mean())

# predict
svc.predict(x_test_scale)

# model evaluation
accuracy_score(svc.predict(x_test_scale), y_test)

# Save the trained model and encoders
joblib.dump(svc, filename="model/model.pkl")
joblib.dump(encoders, "model/encoders.pkl")