import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

# load data
df = pd.read_csv("dataset_housing_price.csv")

# change binaries 'yes'/'no' to 1/0
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].replace({'yes': 1, 'no': 0})

# separate
X = df.drop(columns=['price'])
y = df['price']

# make sure all values are strings
X['furnished'] = X['furnished'].astype(str)

# preprocesor furnished
preprocessor = ColumnTransformer(
    transformers=[
        ('furnished', OneHotEncoder(handle_unknown='ignore'), ['furnished'])
    ],
    remainder='passthrough'  # keep the other columns as they are 
)

# create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model.fit(X_train, y_train)

# save trained model
joblib.dump(model, 'house_price_model.joblib')
print("Modelo entrenado y guardado correctamente.")
