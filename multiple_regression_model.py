"""
Multiple Linear Regression Model
A co# Data Upload Section
if option == "Data Upload & Overview":
    st.header("📂 Data Upload & Overview")
    
    # Option to use sample dataset
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    with col2:
        if st.button("📊 Use Sample Housing Dataset"):
            try:
                st.session_state.df = pd.read_csv('sample_housing_dataset.csv')
                st.success("✅ Sample dataset loaded successfully!")
            except FileNotFoundError:
                st.error("❌ Sample dataset not found. Please upload a CSV file or create the sample dataset first.")
    
    # Handle uploaded fileensive regression analysis tool with multiple algorithms and interactive interface
Author: Nagaraj Satish Navada
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Multiple Regression Model",
    page_icon="📊",
    layout="wide"
)

# Title and Description
st.title("📊 Multiple Linear Regression Model")
st.markdown("### A comprehensive tool for regression analysis with multiple machine learning algorithms")

# Sidebar
st.sidebar.header("📋 Navigation")
option = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["Data Upload & Overview", "Model Training", "Model Comparison", "Predictions"]
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Data Upload Section
if option == "Data Upload & Overview":
    st.header("📁 Data Upload & Overview")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        df = st.session_state.df
        
        st.success(f"✅ Data uploaded successfully! Shape: {df.shape}")
        
        # Data Overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Dataset Info")
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        
        with col2:
            st.subheader("📊 Data Types")
            st.write(df.dtypes.value_counts())
        
        # Display data
        st.subheader("👀 Dataset Preview")
        # Convert to display-friendly format to avoid PyArrow issues
        display_df = df.head().copy()
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str)
        st.dataframe(display_df, use_container_width=True)
        
        # Statistical Summary
        st.subheader("📈 Statistical Summary")
        # Only show numeric columns for statistical summary
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.warning("No numeric columns found for statistical summary.")
        
        # Correlation Matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("🔥 Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation = df[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)

# Model Training Section
elif option == "Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        # Feature Selection
        st.subheader("🎯 Feature Selection")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("❌ Not enough numeric columns for regression analysis!")
        else:
            target_col = st.selectbox("Select Target Variable (y):", numeric_cols)
            feature_cols = st.multiselect(
                "Select Feature Variables (X):",
                [col for col in numeric_cols if col != target_col],
                default=[col for col in numeric_cols if col != target_col][:3]
            )
            
            if len(feature_cols) > 0:
                X = df[feature_cols]
                y = df[target_col]
                
                # Remove missing values
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]
                
                st.success(f"✅ Selected {len(feature_cols)} features and {len(X)} samples")
                
                # Train-Test Split
                st.subheader("🔄 Train-Test Split")
                test_size = st.slider("Test Size:", 0.1, 0.5, 0.2, 0.05)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Feature Scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Model Selection
                st.subheader("🎯 Select Algorithm")
                model_choice = st.selectbox(
                    "Choose Regression Algorithm:",
                    ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest"]
                )
                
                # Train Model
                if st.button("🚀 Train Model"):
                    with st.spinner("Training model..."):
                        if model_choice == "Linear Regression":
                            model = LinearRegression()
                        elif model_choice == "Ridge Regression":
                            alpha = st.sidebar.slider("Ridge Alpha:", 0.1, 10.0, 1.0)
                            model = Ridge(alpha=alpha)
                        elif model_choice == "Lasso Regression":
                            alpha = st.sidebar.slider("Lasso Alpha:", 0.1, 10.0, 1.0)
                            model = Lasso(alpha=alpha)
                        else:  # Random Forest
                            n_estimators = st.sidebar.slider("Number of Trees:", 10, 200, 100)
                            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                        
                        # Use scaled features for linear models, original for tree-based
                        if model_choice in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        # Calculate Metrics
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Display Results
                        st.success("✅ Model trained successfully!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{r2:.4f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:.4f}")
                        with col3:
                            st.metric("MAE", f"{mae:.4f}")
                        with col4:
                            st.metric("MSE", f"{mse:.4f}")
                        
                        # Visualization
                        st.subheader("📈 Model Performance")
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Actual vs Predicted
                        ax1.scatter(y_test, y_pred, alpha=0.6)
                        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                        ax1.set_xlabel('Actual Values')
                        ax1.set_ylabel('Predicted Values')
                        ax1.set_title('Actual vs Predicted Values')
                        
                        # Residuals
                        residuals = y_test - y_pred
                        ax2.scatter(y_pred, residuals, alpha=0.6)
                        ax2.axhline(y=0, color='r', linestyle='--')
                        ax2.set_xlabel('Predicted Values')
                        ax2.set_ylabel('Residuals')
                        ax2.set_title('Residual Plot')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Feature Importance (for tree-based models)
                        if model_choice == "Random Forest":
                            st.subheader("🎯 Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
                            ax.set_title('Feature Importance')
                            st.pyplot(fig)
                        
                        # Save model to session state
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.feature_cols = feature_cols
                        st.session_state.model_choice = model_choice

# Model Comparison Section
elif option == "Model Comparison":
    st.header("🔍 Model Comparison")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            target_col = st.selectbox("Select Target Variable:", numeric_cols)
            feature_cols = st.multiselect(
                "Select Features:",
                [col for col in numeric_cols if col != target_col],
                default=[col for col in numeric_cols if col != target_col][:3]
            )
            
            if len(feature_cols) > 0 and st.button("🔄 Compare All Models"):
                X = df[feature_cols]
                y = df[target_col]
                
                # Remove missing values
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Define models
                models = {
                    'Linear Regression': LinearRegression(),
                    'Ridge Regression': Ridge(alpha=1.0),
                    'Lasso Regression': Lasso(alpha=1.0),
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
                }
                
                results = []
                
                with st.spinner("Comparing models..."):
                    for name, model in models.items():
                        # Use scaled features for linear models
                        if 'Regression' in name:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        results.append({
                            'Model': name,
                            'R² Score': r2,
                            'MSE': mse,
                            'RMSE': np.sqrt(mse),
                            'MAE': mae
                        })
                
                # Display comparison
                results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
                st.subheader("📊 Model Performance Comparison")
                st.dataframe(results_df)
                
                # Visualization
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # R² Score comparison
                sns.barplot(data=results_df, x='R² Score', y='Model', ax=axes[0,0])
                axes[0,0].set_title('R² Score Comparison')
                
                # MSE comparison
                sns.barplot(data=results_df, x='MSE', y='Model', ax=axes[0,1])
                axes[0,1].set_title('MSE Comparison')
                
                # RMSE comparison
                sns.barplot(data=results_df, x='RMSE', y='Model', ax=axes[1,0])
                axes[1,0].set_title('RMSE Comparison')
                
                # MAE comparison
                sns.barplot(data=results_df, x='MAE', y='Model', ax=axes[1,1])
                axes[1,1].set_title('MAE Comparison')
                
                plt.tight_layout()
                st.pyplot(fig)

# Predictions Section
elif option == "Predictions":
    st.header("🔮 Make Predictions")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model first!")
    else:
        st.subheader("📝 Input Features")
        
        feature_cols = st.session_state.feature_cols
        model = st.session_state.model
        scaler = st.session_state.scaler
        model_choice = st.session_state.model_choice
        
        # Input features
        input_data = {}
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(feature_cols):
            with col1 if i % 2 == 0 else col2:
                input_data[feature] = st.number_input(f"Enter {feature}:", value=0.0)
        
        if st.button("🎯 Make Prediction"):
            # Prepare input
            input_df = pd.DataFrame([input_data])
            
            # Scale if needed
            if model_choice in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
            else:
                prediction = model.predict(input_df)[0]
            
            st.success(f"🎯 Predicted Value: **{prediction:.4f}**")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Author:** Nagaraj Satish Navada")
st.sidebar.markdown("**Project:** Multiple Regression Model")
st.sidebar.markdown("**Framework:** Streamlit + Scikit-learn")