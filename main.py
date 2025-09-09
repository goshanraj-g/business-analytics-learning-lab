import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data_file_path = 'data/injury_data.csv'
injury_data = pd.read_csv(data_file_path)
print(injury_data.columns)

y = injury_data.Likelihood_of_Injury

injury_features = ["Player_Age", "Player_Weight", "Player_Height",
                   "Previous_Injuries", "Training_Intensity", "Recovery_Time"]

x = injury_data[injury_features]

# print(x.describe())
# print(x.head())


#####

injury_model = DecisionTreeRegressor(random_state=1)
injury_model.fit(x, y)

print("Predictions for the first 5 houses")
print(x.head())
print("Predictions:")
print(injury_model.predict(x.head()))
