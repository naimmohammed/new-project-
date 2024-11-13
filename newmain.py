import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Streamlit configuration for display
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Customer Segmentation Analysis")

# Step 1: Load Data
@st.cache
def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

# Specify the path to the data file (update the path as needed)
data_path = "C:\\Users\\gargs\\OneDrive\\Desktop\\customer segmentation\\customer_data.csv"
customer_data = load_data(data_path)

# Step 2: Data Cleaning
customer_data['age'].fillna(customer_data['age'].median(), inplace=True)

# Step 3: Exploratory Data Analysis with Streamlit
st.subheader("Exploratory Data Analysis")

# Age Distribution
st.write("### Age Distribution of Customers")
plt.figure(figsize=(10, 5))
sns.histplot(customer_data['age'], bins=30, kde=True, color='skyblue')
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Frequency")
st.pyplot()

# Gender Distribution
st.write("### Gender Distribution of Customers")
plt.figure(figsize=(6, 5))
sns.countplot(data=customer_data, x='gender')
plt.title("Gender Distribution of Customers")
plt.xlabel("Gender")
plt.ylabel("Count")
st.pyplot()

# Preferred Payment Method
st.write("### Preferred Payment Method")
plt.figure(figsize=(8, 5))
sns.countplot(data=customer_data, x='payment_method')
plt.title("Preferred Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Count")
plt.xticks(rotation=45)
st.pyplot()

# Step 4: Customer Segmentation using KMeans
st.subheader("Customer Segmentation")

# Encoding categorical features (Gender, Payment Method)
label_encoder = LabelEncoder()
customer_data['gender_encoded'] = label_encoder.fit_transform(customer_data['gender'])
customer_data['payment_encoded'] = label_encoder.fit_transform(customer_data['payment_method'])

# Selecting features for clustering
X = customer_data[['age', 'gender_encoded', 'payment_encoded']]

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(X_scaled)

# Displaying Cluster Information
st.write("### Cluster Centers (Standardized):")
st.write(kmeans.cluster_centers_)

st.write("### Cluster Sizes:")
st.write(customer_data['cluster'].value_counts())

# Visualization of Clusters
st.write("### Customer Segmentation Clusters")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=customer_data['age'], y=customer_data['gender_encoded'],
                hue=customer_data['cluster'], palette='viridis', s=60)
plt.title("Customer Segmentation Clusters")
plt.xlabel("Age")
plt.ylabel("Gender (Encoded)")
plt.legend(title="Cluster")
st.pyplot()

# Optional: Display customer data
if st.checkbox("Show raw customer data"):
    st.write(customer_data)
