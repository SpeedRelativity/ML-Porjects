
# Step 1: Importing data and reading the required rows and columns from the csv file.
import pandas as pd

# Im importing data I scraped from realtor.com which has a bunch of information about housing prices based on details.
data1 = pd.read_csv('realtor.csv')
# Drop columns starting from the 5th column onward
data1 = data1.drop(columns=data1.columns[4:])


# Step 2: Cleaning the data, removing $ and , and also empty rows from the data.
data1['Price'] = data1['Price'].str.replace('$','').str.replace(',','')
data1['Square_feet'] = data1['Square_feet'].str.replace(',','')

data1 = data1.apply(pd.to_numeric, errors='coerce') #convert everything to numbers.
data1 = data1.dropna(subset=['Price', 'Square_feet','Bathrooms','Bedrooms']) # remove the row if a cell is empty/

#Price was written as a stirng so converting everything to numbers.
print(data1.isnull().sum())  # Check for missing values

# Printing to check if it worked. It did.
#print(data1[data1.columns[:4]])

# Saving the data in a new file.
data1.to_csv('realtor_cleaned.csv', index=False)
data = pd.read_csv('realtor_cleaned.csv')
print('Data Saved')

# import numpy as np

# number_of_synthetic_data = 100

# synthetic_data = pd.DataFrame({
#     'Bedrooms': np.random.randint(2,6, size=number_of_synthetic_data),
#     'Bathrooms': np.random.randint(1,5, size=number_of_synthetic_data),
#     'Square_feet': np.random.randint(800,2700, size=number_of_synthetic_data),
#     'Price': np.random.randint(400000,1400000,size=number_of_synthetic_data)
# })

# data_combined = pd.concat([data, synthetic_data], ignore_index=True)

# Step 3: Calculating some basic values.
mean_price = data['Price'].mean()
max_price = data['Price'].max()
min_price = data['Price'].min()

# Step 4: Visualizing the data as a line graph.
import matplotlib.pyplot as plt

# Scatter plot for Square_feet vs Price
plt.scatter(data1['Square_feet'], data1['Price'], color='red', label='Square Feet')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Price vs Square Feet')
plt.grid(True)
plt.show()

# Scatter plot for Bedrooms vs Price
plt.scatter(data1['Bedrooms'], data1['Price'], color='green', label='Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.title('Price vs Bedrooms')
plt.grid(True)
plt.show()

# Scatter plot for Bathrooms vs Price
plt.scatter(data1['Bathrooms'], data1['Price'], color='blue', label='Bathrooms')
plt.xlabel('Bathrooms')
plt.ylabel('Price')
plt.title('Price vs Bathrooms')
plt.grid(True)
plt.show()

plt.grid(True) # Show grid lines.
plt.savefig('price_vs_details.png') # Once we are setup with the visualization, we can save that image.



#import seaborn as sns
#sns.pairplot(data, x_vars=['Square_feet', 'Bedrooms','Batharooms'], y_vars=['Price'], kind='reg')
#plt.show()


# Lets Make a Prediction model now.
# First we need to import the scikit-learn library
from sklearn.model_selection import train_test_split #used to split our training and testing data
from sklearn.linear_model import LinearRegression # The Linear Regression model
from sklearn.metrics import mean_squared_error, r2_score # For evaluating the model's performance


# Next, we split our data into features (X) and target (y)
x = data[['Bedrooms','Bathrooms','Square_feet']]
y = data['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1) #random state can be any number its just a seed.

# Here's a basic model that learns.
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions using the testing set

y_pred = model.predict(x_test)
print(f'The y prediction is: {y_pred}')

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f' The MSE is: {mse:.2f}')
print(f' The r2 is: {r2}')

#print(data.describe())

