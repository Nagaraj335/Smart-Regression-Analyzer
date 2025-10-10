import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    BayesianRidge, HuberRegressor, SGDRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Comprehensive Regression Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .algorithm-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚀 Comprehensive Regression Suite</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">From Linear to Neural Networks - All Regression Algorithms in One Place</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Configuration Panel")

# Regression Algorithms Dictionary
ALGORITHMS = {
    "Linear Models": {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Elastic Net": ElasticNet(alpha=1.0, l1_ratio=0.5),
        "Bayesian Ridge": BayesianRidge(),
        "Huber Regressor": HuberRegressor(),
        "SGD Regressor": SGDRegressor(random_state=42)
    },
    "Tree-Based Models": {
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42)
    },
    "Instance-Based": {
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Support Vector Regression": SVR(kernel='rbf', C=1.0)
    },
    "Neural Networks": {
        "Multi-Layer Perceptron": MLPRegressor(hidden_layer_sizes=(100,), random_state=42, max_iter=500)
    }
}

# Algorithm Information
ALGORITHM_INFO = {
    "Linear Regression": "Basic linear relationship: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ",
    "Ridge Regression": "Linear regression with L2 regularization to prevent overfitting",
    "Lasso Regression": "Linear regression with L1 regularization for feature selection",
    "Elastic Net": "Combines Ridge and Lasso regularization (L1 + L2)",
    "Bayesian Ridge": "Bayesian approach to ridge regression with automatic relevance determination",
    "Huber Regressor": "Robust to outliers using Huber loss function",
    "SGD Regressor": "Stochastic Gradient Descent for large-scale learning",
    "Decision Tree": "Tree-based model that splits data based on feature values",
    "Random Forest": "Ensemble of decision trees with bootstrap aggregating",
    "Extra Trees": "Extremely randomized trees with random feature selection",
    "Gradient Boosting": "Sequential ensemble that corrects previous model errors",
    "AdaBoost": "Adaptive boosting that focuses on misclassified instances",
    "K-Nearest Neighbors": "Prediction based on k nearest neighbors in feature space",
    "Support Vector Regression": "Uses support vectors to find optimal regression line",
    "Multi-Layer Perceptron": "Neural network with hidden layers for complex patterns"
}

# Step 1: Data Upload
st.markdown('<h2 class="sub-header">📁 Step 1: Data Upload</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
    
    # Display basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(10))
    
    with col2:
        st.subheader("📈 Dataset Statistics")
        st.dataframe(df.describe())
    
    # Missing values check
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.warning("⚠️ Missing values detected:")
        st.dataframe(missing_values[missing_values > 0])
        
        # Option to handle missing values
        handle_missing = st.selectbox(
            "How to handle missing values?",
            ["Drop rows with missing values", "Fill with mean", "Fill with median"]
        )
        
        if handle_missing == "Drop rows with missing values":
            df = df.dropna()
        elif handle_missing == "Fill with mean":
            df = df.fillna(df.mean())
        elif handle_missing == "Fill with median":
            df = df.fillna(df.median())
    
    # Step 2: Feature Selection
    st.markdown('<h2 class="sub-header">🎯 Step 2: Feature Selection</h2>', unsafe_allow_html=True)
    
    # Select target variable
    target_col = st.selectbox("Select Target Variable (y)", df.columns)
    
    # Select feature variables
    feature_cols = st.multiselect(
        "Select Feature Variables (X)", 
        [col for col in df.columns if col != target_col],
        default=[col for col in df.columns if col != target_col][:5]
    )
    
    if len(feature_cols) > 0:
        X = df[feature_cols]
        y = df[target_col]
        
        # Display correlation heatmap
        st.subheader("🔥 Correlation Heatmap")
        
        # Filter for numeric columns only for correlation
        numeric_features = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
        
        # Check if target is numeric
        target_is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
        
        if len(numeric_features) > 0 and target_is_numeric:
            corr_data = df[numeric_features + [target_col]].corr()
        elif len(numeric_features) > 0:
            corr_data = df[numeric_features].corr()
            st.warning("Target variable is not numeric, showing correlations between features only.")
        else:
            st.warning("No numeric features selected for correlation analysis.")
            corr_data = None
        
        if corr_data is not None:
            fig = px.imshow(
                corr_data,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Step 3: Algorithm Selection and Configuration
        st.markdown('<h2 class="sub-header">🤖 Step 3: Algorithm Selection</h2>', unsafe_allow_html=True)
        
        # Multi-algorithm selection
        selected_algorithms = []
        
        for category, algorithms in ALGORITHMS.items():
            st.subheader(f"📂 {category}")
            
            for alg_name, alg_model in algorithms.items():
                col1, col2 = st.columns([3, 7])
                
                with col1:
                    if st.checkbox(alg_name, key=f"check_{alg_name}"):
                        selected_algorithms.append((alg_name, alg_model))
                
                with col2:
                    st.info(ALGORITHM_INFO[alg_name])
        
        # Data splitting
        st.markdown('<h2 class="sub-header">📊 Step 4: Data Preparation</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        with col2:
            random_state = st.number_input("Random State", 0, 1000, 42)
        
        with col3:
            scale_features = st.checkbox("Scale Features", value=True)
        
        # Polynomial features option
        poly_features = st.checkbox("Add Polynomial Features (degree 2)")
        
        # Handle non-numeric features
        st.subheader("🔧 Data Preprocessing")
        
        # Identify numeric and non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if len(non_numeric_cols) > 0:
            st.warning(f"Found non-numeric columns: {non_numeric_cols}")
            handle_categorical = st.selectbox(
                "How to handle categorical variables?",
                ["Remove categorical columns", "One-hot encode", "Label encode"]
            )
            
            if handle_categorical == "Remove categorical columns":
                X = X[numeric_cols]
                st.success(f"Removed {len(non_numeric_cols)} categorical columns. Using {len(numeric_cols)} numeric features.")
            
            elif handle_categorical == "One-hot encode":
                # One-hot encode categorical variables
                X_encoded = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)
                X = X_encoded
                st.success(f"One-hot encoded {len(non_numeric_cols)} categorical columns. Total features: {X.shape[1]}")
            
            elif handle_categorical == "Label encode":
                # Label encode categorical variables
                from sklearn.preprocessing import LabelEncoder
                X_encoded = X.copy()
                for col in non_numeric_cols:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                X = X_encoded
                st.success(f"Label encoded {len(non_numeric_cols)} categorical columns.")
        
        # Ensure target variable is numeric
        if not pd.api.types.is_numeric_dtype(y):
            st.error("Target variable must be numeric for regression. Please select a numeric target variable.")
            st.stop()
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if poly_features:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train = poly.fit_transform(X_train)
            X_test = poly.transform(X_test)
        
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Step 5: Model Training and Evaluation
        if len(selected_algorithms) > 0:
            st.markdown('<h2 class="sub-header">🚀 Step 5: Model Training & Results</h2>', unsafe_allow_html=True)
            
            results = []
            trained_models = {}
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (alg_name, alg_model) in enumerate(selected_algorithms):
                status_text.text(f'Training {alg_name}...')
                
                try:
                    # Train model
                    alg_model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred_train = alg_model.predict(X_train)
                    y_pred_test = alg_model.predict(X_test)
                    
                    # Metrics
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    
                    results.append({
                        'Algorithm': alg_name,
                        'Train R²': round(train_r2, 4),
                        'Test R²': round(test_r2, 4),
                        'Train MAE': round(train_mae, 4),
                        'Test MAE': round(test_mae, 4),
                        'Train RMSE': round(train_rmse, 4),
                        'Test RMSE': round(test_rmse, 4),
                        'Overfitting': round(train_r2 - test_r2, 4)
                    })
                    
                    trained_models[alg_name] = {
                        'model': alg_model,
                        'y_pred_train': y_pred_train,
                        'y_pred_test': y_pred_test
                    }
                    
                except Exception as e:
                    st.error(f"Error training {alg_name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(selected_algorithms))
            
            status_text.text('Training complete!')
            
            # Results DataFrame
            results_df = pd.DataFrame(results)
            
            # Display results table
            st.subheader("📊 Model Performance Comparison")
            st.dataframe(
                results_df.style.highlight_max(subset=['Test R²'], color='lightgreen')
                                .highlight_min(subset=['Test MAE', 'Test RMSE'], color='lightgreen')
                                .format({'Train R²': '{:.4f}', 'Test R²': '{:.4f}',
                                       'Train MAE': '{:.4f}', 'Test MAE': '{:.4f}',
                                       'Train RMSE': '{:.4f}', 'Test RMSE': '{:.4f}',
                                       'Overfitting': '{:.4f}'})
            )
            
            # Best model identification
            best_model_name = results_df.loc[results_df['Test R²'].idxmax(), 'Algorithm']
            st.success(f"🏆 Best performing model: **{best_model_name}** (Test R² = {results_df['Test R²'].max():.4f})")
            
            # Visualization Section
            st.markdown('<h2 class="sub-header">📈 Step 6: Visualizations</h2>', unsafe_allow_html=True)
            
            # Performance comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    results_df, 
                    x='Algorithm', 
                    y='Test R²',
                    title='Test R² Score Comparison',
                    color='Test R²',
                    color_continuous_scale='viridis'
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    results_df, 
                    x='Algorithm', 
                    y='Test RMSE',
                    title='Test RMSE Comparison (Lower is Better)',
                    color='Test RMSE',
                    color_continuous_scale='viridis_r'
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis for best model
            st.subheader(f"🔍 Detailed Analysis: {best_model_name}")
            
            best_model_data = trained_models[best_model_name]
            
            # Prediction vs Actual plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_train, 
                    y=best_model_data['y_pred_train'],
                    mode='markers',
                    name='Training Data',
                    opacity=0.6
                ))
                fig.add_trace(go.Scatter(
                    x=[y_train.min(), y_train.max()],
                    y=[y_train.min(), y_train.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title='Training: Predicted vs Actual',
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test, 
                    y=best_model_data['y_pred_test'],
                    mode='markers',
                    name='Test Data',
                    opacity=0.6
                ))
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title='Testing: Predicted vs Actual',
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Residuals analysis
            st.subheader("📊 Residuals Analysis")
            
            residuals_train = y_train - best_model_data['y_pred_train']
            residuals_test = y_test - best_model_data['y_pred_test']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=best_model_data['y_pred_train'],
                    y=residuals_train,
                    mode='markers',
                    name='Training Residuals',
                    opacity=0.6
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    title='Training Residuals Plot',
                    xaxis_title='Predicted Values',
                    yaxis_title='Residuals'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=best_model_data['y_pred_test'],
                    y=residuals_test,
                    mode='markers',
                    name='Test Residuals',
                    opacity=0.6
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    title='Test Residuals Plot',
                    xaxis_title='Predicted Values',
                    yaxis_title='Residuals'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (for tree-based models)
            if hasattr(trained_models[best_model_name]['model'], 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                if poly_features:
                    feature_names = [f"Feature_{i}" for i in range(len(trained_models[best_model_name]['model'].feature_importances_))]
                else:
                    feature_names = feature_cols
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': trained_models[best_model_name]['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(10), 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Top 10 Feature Importances'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            st.markdown('<h2 class="sub-header">💾 Step 7: Download Results</h2>', unsafe_allow_html=True)
            
            csv_results = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_results,
                file_name=f"regression_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("⚠️ Please select at least one algorithm to proceed.")
    
    else:
        st.warning("⚠️ Please select at least one feature variable.")

else:
    # Sample data information
    st.markdown('<h2 class="sub-header">🎯 Get Started</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Welcome to the Comprehensive Regression Suite!** 🚀
    
    This application supports **15+ regression algorithms** across multiple categories:
    
    - 📈 **Linear Models**: Linear, Ridge, Lasso, Elastic Net, Bayesian Ridge, Huber, SGD
    - 🌳 **Tree-Based**: Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost  
    - 📊 **Instance-Based**: K-Nearest Neighbors, Support Vector Regression
    - 🧠 **Neural Networks**: Multi-Layer Perceptron
    
    **Features:**
    - Compare multiple algorithms simultaneously
    - Interactive visualizations and residual analysis
    - Feature importance analysis
    - Comprehensive performance metrics
    - Download results as CSV
    
    **To get started:** Upload your CSV file above! 📁
    """)
    
    # Sample data format
    st.subheader("📋 Expected Data Format")
    sample_data = pd.DataFrame({
        'Feature_1': [1.2, 2.3, 3.1, 4.5, 5.2],
        'Feature_2': [10, 15, 20, 25, 30],
        'Feature_3': [0.5, 1.0, 1.5, 2.0, 2.5],
        'Target': [100, 150, 200, 250, 300]
    })
    st.dataframe(sample_data)
    st.caption("Your CSV should have numerical features and one target column.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; margin: 2rem 0;'>
        <p>🚀 <strong>Comprehensive Regression Suite</strong> | Built with Streamlit & Scikit-learn</p>
        <p>Inspired by Giorgio De Simone's Multiple Linear Regression project - Enhanced with 15+ algorithms! 🎯</p>
    </div>
    """, 
    unsafe_allow_html=True
)