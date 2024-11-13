import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure Streamlit page layout
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title and description
st.title("Customer Segmentation Analysis")
st.write("This application performs exploratory data analysis (EDA) and customer segmentation using KMeans clustering.")

# Load the data
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    customer_data = pd.read_csv(uploaded_file)
    
    # Display first few rows of the data
    st.subheader("Data Overview")
    st.write(customer_data.head())
    
    # Fill missing 'age' values with median
    customer_data['age'].fillna(customer_data['age'].median(), inplace=True)

    # Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis")

    # Age Distribution
    st.write("### Age Distribution")
    fig_age, ax = plt.subplots()
    sns.histplot(customer_data['age'], bins=30, kde=True, color='skyblue', ax=ax)
    ax.set_title("Age Distribution of Customers")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig_age)

    # Gender Distribution
    st.write("### Gender Distribution")
    fig_gender, ax = plt.subplots()
    sns.countplot(data=customer_data, x='gender', palette='pastel', ax=ax)
    ax.set_title("Gender Distribution of Customers")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    st.pyplot(fig_gender)

    # Preferred Payment Method
    st.write("### Preferred Payment Method")
    fig_payment, ax = plt.subplots()
    sns.countplot(data=customer_data, x='payment_method', palette='muted', ax=ax)
    ax.set_title("Preferred Payment Method")
    ax.set_xlabel("Payment Method")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig_payment)

    # Customer Segmentation with KMeans
    st.subheader("Customer Segmentation using KMeans")

    # Encoding categorical features
    label_encoder = LabelEncoder()
    customer_data['gender_encoded'] = label_encoder.fit_transform(customer_data['gender'])
    customer_data['payment_encoded'] = label_encoder.fit_transform(customer_data['payment_method'])

    # Selecting features for clustering
    X = customer_data[['age', 'gender_encoded', 'payment_encoded']]

    # Standardizing features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # User selection for number of clusters
    n_clusters = st.slider("Select Number of Clusters for KMeans", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_data['cluster'] = kmeans.fit_predict(X_scaled)

    # Display cluster centers
    st.write("### Cluster Centers (Standardized)")
    st.write(pd.DataFrame(kmeans.cluster_centers_, columns=['Age', 'Gender (Encoded)', 'Payment Method (Encoded)']))

    # Cluster Visualization
    st.write("### Customer Segmentation Clusters")
    fig_cluster, ax = plt.subplots()
    sns.scatterplot(x=customer_data['age'], y=customer_data['gender_encoded'],
                    hue=customer_data['cluster'], palette='viridis', s=60, ax=ax)
    ax.set_title("Customer Segmentation Clusters")
    ax.set_xlabel("Age")
    ax.set_ylabel("Gender (Encoded)")
    st.pyplot(fig_cluster)

    # Display cluster sizes
    st.write("### Cluster Sizes")
    st.write(customer_data['cluster'].value_counts())
else:
    st.write("Please upload a CSV file to start the analysis.")