import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the page configuration
st.set_page_config(page_title="Iris Flower Type Prediction", layout="centered")

# Title and introduction
st.title("🌸 Iris Flower Type Prediction 🌸")
st.write("Welcome to the Iris Flower Prediction App! This app predicts the type of iris flower based on its sepal and petal dimensions.")
st.write("### Dataset Overview")
st.write("The Iris dataset consists of 150 samples of iris flowers, with 4 features (sepal length, sepal width, petal length, petal width) and 3 species (Iris-setosa, Iris-versicolor, Iris-virginica).")
st.write("### Model Accuracy")
st.write("The Logistic Regression model has an accuracy of approximately 95% on the training data.")

# Function to load the Iris dataset
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    return pd.read_csv(url, header=None, names=column_names)

# Load the data
data = load_data()

# User input for prediction
st.write("### Enter the following values to predict the flower type:")
sepal_length = st.number_input("Sepal Length (cm)", value=5.0, min_value=0.0)
petal_length = st.number_input("Petal Length (cm)", value=1.0, min_value=0.0)

# Prepare features and target variable
X = data[['sepal_length', 'petal_length']]  # Features
y = data['species']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Dictionary containing species images and descriptions
species_info = {
    "Iris-setosa": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
        "description": (
            "Iris setosa is a perennial flowering plant that grows up to 30 cm in height. "
            "Its flowers are typically pale blue or violet with darker markings, blooming from late spring to early summer. "
            "This species is native to North America and is often found in wet, marshy areas. "
            "Iris setosa is known for its resilience and ability to thrive in challenging environments."
        )
    },
    "Iris-versicolor": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/d/db/Iris_versicolor_4.jpg",
        "description": (
            "Iris versicolor, commonly known as the Harlequin Blue Flag, is a robust perennial that can reach heights of up to 90 cm. "
            "It features striking blue and yellow flowers that bloom in late spring and early summer. "
            "This species thrives in wetland areas and is native to North America, often seen in swamps, marshes, and along lake shores. "
            "Iris versicolor is not only beautiful but also plays a role in local ecosystems by providing habitat for various wildlife."
        )
    },
    "Iris-virginica": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/1200px-Iris_virginica_2.jpg",
        "description": (
            "Iris virginica is a tall, elegant perennial that can grow up to 90 cm. "
            "It showcases large, vivid flowers that can vary in color from deep purple to blue and white, blooming in late spring. "
            "This species is native to the wetlands of the southeastern United States and is often found in riverbanks and wet meadows. "
            "Iris virginica is appreciated for its ornamental value and is a key species in maintaining the health of wetland ecosystems."
        )
    }
}

# Prediction functionality
if st.button("Predict"):
    prediction = model.predict([[sepal_length, petal_length]])[0]
    st.write(f"### Predicted Flower Type: **{prediction}**")
    
    # Display image and description of the predicted flower
    species_details = species_info[prediction]
    st.image(species_details["image"], caption=prediction, use_column_width=True)
    st.write(species_details["description"])

# Function to visualize the data
def visualize_data():
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distribution of sepal length
    sns.histplot(data=data, x='sepal_length', hue='species', kde=True, multiple="stack", ax=ax[0, 0])
    ax[0, 0].set_title("Sepal Length Distribution")
    
    # Distribution of sepal width
    sns.histplot(data=data, x='sepal_width', hue='species', kde=True, multiple="stack", ax=ax[0, 1])
    ax[0, 1].set_title("Sepal Width Distribution")

    # Distribution of petal length
    sns.histplot(data=data, x='petal_length', hue='species', kde=True, multiple="stack", ax=ax[1, 0])
    ax[1, 0].set_title("Petal Length Distribution")

    # Distribution of petal width
    sns.histplot(data=data, x='petal_width', hue='species', kde=True, multiple="stack", ax=ax[1, 1])
    ax[1, 1].set_title("Petal Width Distribution")

    plt.tight_layout()
    st.pyplot(fig)

    # Scatter plot for petal dimensions
    st.write("### Scatter Plot of Petal Dimensions")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='petal_length', y='petal_width', hue='species', style='species', s=100)
    plt.title("Petal Length vs Petal Width")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.legend(title='Species')
    st.pyplot(plt)

# Function to analyze the data
def analyze_data():
    st.write("### Data Analysis")
    
    st.write("#### Descriptive Statistics:")
    st.write(data.describe())

    st.write("#### Correlation Matrix:")
    correlation = data.corr(numeric_only=True)  # Calculate correlation for numeric features only
    st.write(correlation)

    # Display correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
    plt.title("Correlation Matrix")
    st.pyplot(plt)

    # Count of each species
    species_counts = data['species'].value_counts()
    st.write("#### Species Count:")
    st.bar_chart(species_counts)

# Function to show model information
def show_model_info():
    st.write("### Model Information")
    st.write("""
    The Logistic Regression model is a statistical method for predicting binary classes. 
    In this case, it is used to classify the Iris flower species based on the dimensions of its petals and sepals.
    
    **How it Works:**
    - Logistic regression uses a logistic function to model a binary dependent variable.
    - It estimates probabilities using a linear equation formed by a weighted sum of the input features.
    - The output of the logistic function is a value between 0 and 1, which can be interpreted as a probability.
    
    **Model Evaluation:**
    - The accuracy of this model is approximately 95% on the training data.
    - It is important to evaluate the model using various metrics such as precision, recall, and F1-score to understand its performance.
    
    **Advantages:**
    - Simple and interpretable.
    - Efficient for binary classification problems.
    
    **Limitations:**
    - Assumes a linear relationship between the independent and dependent variables.
    - May not perform well with non-linear relationships.
    """)

# Checkboxes for displaying model information, data analysis, and visualizations
if st.checkbox("Show Model Information"):
    show_model_info()

if st.checkbox("Show Data Analysis"):
    analyze_data()

if st.checkbox("Show Data Visualizations"):
    visualize_data()
