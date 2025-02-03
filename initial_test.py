from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import pandas as pd

df = pd.read_excel('./data/excel_files/features1-31-2025.xlsx')

# Define target and features
target = df['Volume']
features = df.drop(['Volume', 'UniqueID'], axis=1)

# Preprocessing
transform = ColumnTransformer(transformers=[
    ('One Hot', OneHotEncoder(handle_unknown='ignore'), ["roadclass", "Land Usage"]),
    ("Standard Scale", StandardScaler(), ['Speed (km/h)', 'Lat', 'Long'])
])

x = transform.fit_transform(features)
y = target.to_numpy()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale target variable
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

# Train model
reg = SVR()
reg.fit(x_train, y_train_scaled)

# Predict and inverse transform
pred_scaled = reg.predict(x_test)
pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

# Evaluate
print("R² Score:", r2_score(y_test, pred))
