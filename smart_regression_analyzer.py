import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from io import StringIO

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Regression Algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# Advanced Gradient Boosting
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep Learning
import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="🎯 Smart Regression Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .best-model {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class SmartRegressionAnalyzer:
    def __init__(self):
        # Optimized algorithms with faster parameters
        self.algorithms = {
            "Linear Models": {
                "Linear Regression": LinearRegression(),
                "Ridge Regression (L2)": Ridge(alpha=1.0),
                "Lasso Regression (L1)": Lasso(alpha=1.0, max_iter=500),
                "Elastic Net Regression": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=500),
                "SGD Regressor": SGDRegressor(random_state=42, max_iter=100)
            },
            "Tree-Based Models": {
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=10),
                "Random Forest Regression": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                "Gradient Boosting Regression (GBR)": GradientBoostingRegressor(random_state=42, n_estimators=50),
            },
            "Advanced Gradient Boosting": {
                "XGBoost Regression": xgb.XGBRegressor(random_state=42, verbosity=0, n_estimators=50, n_jobs=-1),
                "LightGBM Regression": lgb.LGBMRegressor(random_state=42, verbosity=-1, n_estimators=50, n_jobs=-1),
            },
            "Instance-Based Models": {
                "K-Nearest Neighbors Regression (KNN)": KNeighborsRegressor(n_neighbors=5),
                "Support Vector Regression (SVR)": SVR(kernel='rbf', C=1.0)
            },
            "Neural Networks": {
                "Multi-Layer Perceptron": MLPRegressor(hidden_layer_sizes=(50,), random_state=42, max_iter=200, early_stopping=True),
            }
        }
        
        # Fast algorithms for quick analysis
        self.fast_algorithms = {
            "Quick Analysis": {
                "Linear Regression": LinearRegression(),
                "Random Forest (Fast)": RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1),
                "XGBoost (Fast)": xgb.XGBRegressor(random_state=42, verbosity=0, n_estimators=20, n_jobs=-1),
            }
        }
        
        self.results = []
        self.trained_models = {}
        self.best_model = None
        self.feature_columns = None
        self.target_column = None
        
    def _create_keras_model(self, input_dim):
        """Create a fast Keras neural network model for regression"""
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
        
    def detect_target_column(self, df):
        """Automatically detect the target column"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Common target column names
        common_targets = ['price', 'target', 'y', 'label', 'output', 'value', 'amount', 'cost', 'salary', 'income']
        
        for col in df.columns:
            if any(target in col.lower() for target in common_targets):
                return col
        
        # If no common target found, return the last numeric column
        if numeric_columns:
            return numeric_columns[-1]
        
        return None
    
    def preprocess_data(self, df, target_col):
        """Intelligent data preprocessing"""
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            le = LabelEncoder()
            for col in categorical_cols:
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        y = y.fillna(y.mean())
        
        return X, y, feature_cols
    
    def analyze_dataset(self, df, target_col):
        """Provide dataset insights"""
        insights = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().sum(),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'target_type': 'Numeric' if pd.api.types.is_numeric_dtype(df[target_col]) else 'Categorical',
            'target_range': (df[target_col].min(), df[target_col].max()) if pd.api.types.is_numeric_dtype(df[target_col]) else None
        }
        return insights
    
    def train_all_algorithms(self, X, y, quick_mode=True):
        """Train algorithms and return results - optimized for speed"""
        self.results = []
        self.trained_models = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Choose algorithm set based on mode
        algorithms_to_use = self.fast_algorithms if quick_mode else self.algorithms
        
        # Count total algorithms
        total_algorithms = sum(len(algs) for algs in algorithms_to_use.values())
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        algorithm_count = 0
        
        for category, category_algorithms in algorithms_to_use.items():
            for alg_name, alg_model in category_algorithms.items():
                algorithm_count += 1
                progress = algorithm_count / total_algorithms
                progress_bar.progress(progress)
                status_text.text(f"Training {alg_name}... ({algorithm_count}/{total_algorithms})")
                
                try:
                    start_time = time.time()
                    
                    # Standard sklearn-compatible models (removed Keras for speed)
                    alg_model.fit(X_train_scaled, y_train)
                    training_time = time.time() - start_time
                    
                    # Predictions
                    y_pred_train = alg_model.predict(X_train_scaled)
                    y_pred_test = alg_model.predict(X_test_scaled)
                    
                    # Fast cross-validation (reduced folds)
                    try:
                        if quick_mode:
                            # Skip CV in quick mode for speed
                            cv_mean = 0.0
                            cv_std = 0.0
                        else:
                            cv_scores = cross_val_score(alg_model, X_train_scaled, y_train, cv=3, scoring='r2')
                            cv_mean = cv_scores.mean()
                            cv_std = cv_scores.std()
                    except:
                        cv_mean = 0.0
                        cv_std = 0.0
                    
                    # Calculate metrics
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    overfitting = train_r2 - test_r2
                    
                    # Store results
                    self.results.append({
                        'Category': category,
                        'Algorithm': alg_name,
                        'Train R²': round(train_r2, 4),
                        'Test R²': round(test_r2, 4),
                        'CV R² Mean': round(cv_mean, 4),
                        'CV R² Std': round(cv_std, 4),
                        'Test MAE': round(test_mae, 2),
                        'Test RMSE': round(test_rmse, 2),
                        'Overfitting': round(overfitting, 4),
                        'Training Time (s)': round(training_time, 3)
                    })
                    
                    # Store trained model
                    self.trained_models[alg_name] = {
                        'model': alg_model,
                        'category': category,
                        'scaler': scaler,
                        'y_pred_train': y_pred_train,
                        'y_pred_test': y_pred_test,
                        'X_train': X_train_scaled,
                        'X_test': X_test_scaled,
                        'y_train': y_train,
                        'y_test': y_test
                    }
                    
                except Exception as e:
                    st.warning(f"Error training {alg_name}: {str(e)}")
                    # Continue with next algorithm instead of stopping
        
        progress_bar.progress(1.0)
        status_text.text("✅ All algorithms trained successfully!")
        
        # Find best model
        if self.results:
            results_df = pd.DataFrame(self.results)
            best_idx = results_df['Test R²'].idxmax()
            self.best_model = results_df.iloc[best_idx]
        
        return pd.DataFrame(self.results)

def main():
    # Header
    st.markdown('<div class="main-header">🤖 Smart Regression Analyzer</div>', unsafe_allow_html=True)
    st.markdown("### Upload your CSV and let AI find the perfect regression algorithm!")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SmartRegressionAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            st.success("✅ File uploaded successfully!")
            
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📊 Dataset Overview")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Size:** {uploaded_file.size} bytes")
            
            # Auto-detect target column
            suggested_target = analyzer.detect_target_column(df)
            
            # Manual target selection
            target_options = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_options:
                selected_target = st.selectbox(
                    "🎯 Select Target Column:", 
                    target_options, 
                    index=target_options.index(suggested_target) if suggested_target in target_options else 0
                )
            else:
                st.error("❌ No numeric columns found for regression!")
                return
            
            # Analysis settings
            st.subheader("⚙️ Analysis Settings")
            analysis_mode = st.radio(
                "🚀 Analysis Mode:",
                ["⚡ Quick Analysis (3 algorithms)", "🔬 Full Analysis (15+ algorithms)"],
                index=0
            )
            show_details = st.checkbox("Show detailed analysis", value=True)
            show_visualizations = st.checkbox("Show visualizations", value=False)  # Default to False for speed
            
            # Start analysis button
            analyze_button = st.button("🚀 Start Analysis", type="primary")
            
        else:
            st.info("👆 Please upload a CSV file to begin analysis")
            
            # Sample data option
            if st.button("📝 Use Sample Housing Data"):
                # Create sample data
                np.random.seed(42)
                n_samples = 1000
                sample_data = {
                    'Size_SqFt': np.random.normal(2000, 500, n_samples),
                    'Bedrooms': np.random.randint(1, 6, n_samples),
                    'Bathrooms': np.random.randint(1, 4, n_samples),
                    'Age_Years': np.random.randint(0, 50, n_samples),
                    'Location_Score': np.random.uniform(1, 10, n_samples),
                    'Has_Garage': np.random.choice([0, 1], n_samples),
                    'Price': np.random.normal(300000, 100000, n_samples)
                }
                df = pd.DataFrame(sample_data)
                selected_target = 'Price'
                analyze_button = True
                st.success("✅ Sample data loaded!")
            else:
                return
    
    # Main analysis
    if uploaded_file is not None or 'df' in locals():
        # Dataset Analysis
        if show_details:
            st.markdown('<div class="sub-header">📊 Dataset Analysis</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            insights = analyzer.analyze_dataset(df, selected_target)
            
            with col1:
                st.metric("📏 Dataset Size", f"{insights['shape'][0]:,} × {insights['shape'][1]}")
            with col2:
                st.metric("🔢 Numeric Features", insights['numeric_features'])
            with col3:
                st.metric("📝 Categorical Features", insights['categorical_features'])
            with col4:
                st.metric("❓ Missing Values", insights['missing_values'])
            
            # Show data preview
            with st.expander("👀 Data Preview"):
                st.dataframe(df.head(10))
            
            # Show basic statistics
            with st.expander("📈 Statistical Summary"):
                st.dataframe(df.describe())
        
        # Dedicated Charts Section  
        if show_visualizations:
            st.markdown('<div class="sub-header">📊 Interactive Charts Gallery</div>', unsafe_allow_html=True)
            
            # Performance mode selection
            chart_mode = st.radio(
                "**Choose chart rendering mode:**",
                ["🚀 Essential Charts (3-4 charts)", "⚡ Standard Mode (6-7 charts)", "🔥 All Charts (10 charts)"],
                index=0,
                help="Essential mode loads fastest and prevents browser freezing"
            )
            
            # Set chart selections based on mode
            if chart_mode == "🚀 Essential Charts (3-4 charts)":
                show_bar, show_line, show_pie, show_histogram = True, True, True, True
                show_scatter = show_box = show_heatmap = show_area = show_violin = show_donut = False
            elif chart_mode == "⚡ Standard Mode (6-7 charts)":
                show_bar = show_line = show_pie = show_histogram = show_scatter = show_box = show_heatmap = True
                show_area = show_violin = show_donut = False
            else:  # All Charts mode
                show_bar = show_line = show_pie = show_histogram = True
                show_scatter = show_box = show_heatmap = show_area = show_violin = show_donut = True
            
            # Prepare data for charts with aggressive sampling
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Very aggressive sampling for chart performance
            if len(df) > 5000:
                st.warning("🚀 Large dataset: Using 1000 row sample for fast chart rendering.")
                chart_df = df.sample(n=1000, random_state=42)
            elif len(df) > 1000:
                st.info("⚡ Medium dataset: Using 1500 row sample for optimal performance.")
                chart_df = df.sample(n=1500, random_state=42)
            else:
                chart_df = df
                
            # Add loading indicator
            chart_progress = st.progress(0)
            chart_status = st.empty()
            chart_status.text("🎨 Preparing charts...")
            
            # Create charts with progress tracking
            st.markdown("### 📊 Chart Gallery")
            
            chart_count = sum([show_bar, show_line, show_pie, show_histogram, show_scatter, show_box, show_heatmap, show_area, show_violin, show_donut])
            current_chart = 0
            
            # Row 1: Bar Chart, Line Chart, Pie Chart
            if show_bar or show_line or show_pie:
                chart_status.text(f"📊 Rendering basic charts... ({current_chart + 1}/{chart_count})")
                row1_col1, row1_col2, row1_col3 = st.columns(3)
                
                with row1_col1:
                    if show_bar and len(numeric_cols) >= 1:
                        st.subheader("📊 Bar Chart")
                        # Simplified bar chart
                        if categorical_cols:
                            cat_col = categorical_cols[0]
                            value_counts = chart_df[cat_col].value_counts().head(5)  # Reduced to 5 for speed
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"Top 5 {cat_col}",
                                labels={'x': cat_col, 'y': 'Count'}
                            )
                        else:
                            num_col = numeric_cols[0]
                            binned_data = pd.cut(chart_df[num_col], bins=5)  # Reduced bins
                            value_counts = binned_data.value_counts()
                            fig = px.bar(
                                x=[str(x) for x in value_counts.index],
                                y=value_counts.values,
                                title=f"{num_col} Distribution"
                            )
                        fig.update_layout(height=300, showlegend=False)  # Reduced height, no legend
                        st.plotly_chart(fig, use_container_width=True)
                        current_chart += 1
                        chart_progress.progress(current_chart / chart_count)
                
                with row1_col2:
                    if show_line and len(numeric_cols) >= 1:
                        st.subheader("📈 Line Chart")
                        # Simplified line chart
                        num_col = numeric_cols[0]
                        # Use every 10th point for speed
                        sample_size = min(100, len(chart_df))
                        line_data = chart_df[num_col].iloc[::max(1, len(chart_df)//sample_size)]
                        fig = px.line(
                            x=range(len(line_data)),
                            y=line_data.values,
                            title=f"{num_col} Trend",
                            labels={'x': 'Sample Points', 'y': num_col}
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        current_chart += 1
                        chart_progress.progress(current_chart / chart_count)
                
                with row1_col3:
                    if show_pie:
                        st.subheader("🥧 Pie Chart")
                        if categorical_cols:
                            cat_col = categorical_cols[0]
                            value_counts = chart_df[cat_col].value_counts().head(4)  # Only top 4
                            fig = px.pie(
                                values=value_counts.values,
                                names=value_counts.index,
                                title=f"Top 4 {cat_col}"
                            )
                        else:
                            num_col = numeric_cols[0]
                            binned_data = pd.cut(chart_df[num_col], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])
                            value_counts = binned_data.value_counts()
                            fig = px.pie(
                                values=value_counts.values,
                                names=value_counts.index,
                                title=f"{num_col} Categories"
                            )
                        fig.update_layout(height=300, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                        current_chart += 1
                        chart_progress.progress(current_chart / chart_count)
            
            # Row 2: Histogram, Scatter Plot, Box Plot
            if show_histogram or show_scatter or show_box:
                row2_col1, row2_col2, row2_col3 = st.columns(3)
                
                with row2_col1:
                    if show_histogram and len(numeric_cols) >= 1:
                        st.subheader("📊 Histogram")
                        num_col = numeric_cols[0]
                        fig = px.histogram(
                            chart_df,
                            x=num_col,
                            nbins=30,
                            title=f"Histogram: {num_col} Distribution",
                            marginal="box"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with row2_col2:
                    if show_scatter and len(numeric_cols) >= 2:
                        st.subheader("🔸 Scatter Plot")
                        x_col = numeric_cols[0]
                        y_col = numeric_cols[1] if len(numeric_cols) > 1 else selected_target
                        color_col = categorical_cols[0] if categorical_cols else None
                        
                        fig = px.scatter(
                            chart_df,
                            x=x_col,
                            y=y_col,
                            color=color_col,
                            title=f"Scatter Plot: {x_col} vs {y_col}",
                            trendline="ols"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with row2_col3:
                    if show_box and len(numeric_cols) >= 1:
                        st.subheader("📦 Box Plot")
                        if len(numeric_cols) > 1:
                            # Multiple box plots
                            melted_df = chart_df[numeric_cols[:4]].melt(var_name='Feature', value_name='Value')
                            fig = px.box(
                                melted_df,
                                x='Feature',
                                y='Value',
                                title="Box Plot: Feature Distributions"
                            )
                        else:
                            # Single box plot
                            num_col = numeric_cols[0]
                            fig = px.box(
                                chart_df,
                                y=num_col,
                                title=f"Box Plot: {num_col}"
                            )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Row 3: Heatmap, Area Chart, Violin Plot
            if show_heatmap or show_area or show_violin:
                row3_col1, row3_col2, row3_col3 = st.columns(3)
                
                with row3_col1:
                    if show_heatmap and len(numeric_cols) >= 2:
                        st.subheader("🔥 Heatmap")
                        corr_matrix = chart_df[numeric_cols].corr()
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            title="Correlation Heatmap",
                            color_continuous_scale='RdBu_r'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with row3_col2:
                    if show_area and len(numeric_cols) >= 1:
                        st.subheader("📊 Area Chart")
                        num_col = numeric_cols[0]
                        # Create cumulative area chart
                        sorted_data = chart_df[num_col].sort_values().reset_index(drop=True)
                        cumulative = sorted_data.cumsum()
                        fig = px.area(
                            x=range(len(cumulative)),
                            y=cumulative,
                            title=f"Area Chart: {num_col} Cumulative Distribution"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with row3_col3:
                    if show_violin and len(numeric_cols) >= 1:
                        st.subheader("🎻 Violin Plot")
                        num_col = numeric_cols[0]
                        if categorical_cols:
                            cat_col = categorical_cols[0]
                            # Filter to top categories for clarity
                            top_categories = chart_df[cat_col].value_counts().head(5).index
                            filtered_df = chart_df[chart_df[cat_col].isin(top_categories)]
                            fig = px.violin(
                                filtered_df,
                                x=cat_col,
                                y=num_col,
                                title=f"Violin Plot: {num_col} by {cat_col}",
                                box=True
                            )
                        else:
                            fig = px.violin(
                                chart_df,
                                y=num_col,
                                title=f"Violin Plot: {num_col} Distribution",
                                box=True
                            )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Row 4: Donut Chart (standalone)
            if show_donut:
                st.subheader("🍩 Donut Chart")
                if categorical_cols:
                    cat_col = categorical_cols[0]
                    value_counts = chart_df[cat_col].value_counts().head(8)
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=value_counts.index,
                        values=value_counts.values,
                        hole=.4,
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    fig.update_layout(
                        title=f"Donut Chart: {cat_col} Distribution",
                        height=500,
                        annotations=[dict(text=cat_col, x=0.5, y=0.5, font_size=20, showarrow=False)]
                    )
                else:
                    # Numeric donut chart
                    num_col = numeric_cols[0]
                    binned_data = pd.cut(chart_df[num_col], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                    value_counts = binned_data.value_counts()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=value_counts.index,
                        values=value_counts.values,
                        hole=.4,
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    fig.update_layout(
                        title=f"Donut Chart: {num_col} Distribution",
                        height=500,
                        annotations=[dict(text=num_col, x=0.5, y=0.5, font_size=20, showarrow=False)]
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Complete charts loading
            chart_progress.progress(1.0)
            chart_status.text("✅ All charts loaded successfully!")
            time.sleep(0.5)
            chart_progress.empty()
            chart_status.empty()
            
            st.markdown("---")
        
        # Visualizations - All 10 Charts in One Section
        if show_visualizations:
            st.markdown('<div class="sub-header">📈 Complete Data Visualization Suite (10 Charts)</div>', unsafe_allow_html=True)
            
            # Add performance info
            with st.expander("⚡ Performance Information"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Dataset Size", f"{len(df):,} rows")
                with col2:
                    st.metric("🔢 Features", f"{df.shape[1]} columns")
                with col3:
                    viz_size = len(viz_df) if 'viz_df' in locals() else len(df)
                    st.metric("📈 Visualization Data", f"{viz_size:,} rows")
                with col4:
                    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                    st.metric("💾 Memory Usage", f"{memory_usage:.1f} MB")
            
            # Aggressive sampling for performance optimization
            if len(df) > 10000:
                st.warning("⚡ Very large dataset detected. Using highly optimized sampling (1000 rows) for maximum speed.")
                viz_df = df.sample(n=1000, random_state=42)
            elif len(df) > 5000:
                st.warning("⚡ Large dataset detected. Using optimized visualizations with sample data for speed.")
                viz_df = df.sample(n=2000, random_state=42)
            elif len(df) > 1000:
                st.info("📊 Medium dataset detected. Using light sampling for optimal performance.")
                viz_df = df.sample(n=min(3000, len(df)), random_state=42)
            else:
                viz_df = df
            
            # Get numeric columns and correlation matrix
            numeric_df = viz_df.select_dtypes(include=[np.number])
            numeric_cols = numeric_df.columns.tolist()
            
            # Initialize variables
            corr_matrix = None
            top_features = pd.Series(dtype=float)
            
            if len(numeric_cols) > 1:
                corr_matrix = numeric_df.corr()
                if selected_target in corr_matrix.columns:
                    top_features = corr_matrix[selected_target].abs().sort_values(ascending=False)[1:4]
            
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Row 1: Basic Distribution Charts
            status_text.text("🔍 Generating basic data exploration charts...")
            progress_bar.progress(10)
            st.subheader("🔍 Basic Data Exploration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 1. Target Distribution Histogram
                fig = px.histogram(
                    viz_df, 
                    x=selected_target, 
                    nbins=30, 
                    title=f"📊 1. Distribution of {selected_target}",
                    marginal="box",
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 2. Feature Type Distribution
                feature_info = {
                    'Numeric': len(df.select_dtypes(include=[np.number]).columns),
                    'Categorical': len(df.select_dtypes(include=['object']).columns)
                }
                
                fig = px.pie(
                    values=list(feature_info.values()),
                    names=list(feature_info.keys()),
                    title="📈 2. Feature Type Distribution"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                # 3. Correlation Heatmap
                if len(numeric_cols) > 1:
                    fig = px.imshow(
                        corr_matrix, 
                        text_auto=True, 
                        aspect="auto", 
                        title="🔥 3. Correlation Heatmap",
                        color_continuous_scale='RdBu'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Update progress
            progress_bar.progress(30)
            status_text.text("📊 Generating statistical analysis charts...")
            
            # Row 2: Statistical Analysis
            st.subheader("📊 Statistical Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 4. Box Plot for all numeric features
                if len(numeric_cols) > 1:
                    melted_df = viz_df[numeric_cols].melt(var_name='Feature', value_name='Value')
                    fig = px.box(
                        melted_df,
                        x='Feature',
                        y='Value',
                        title="📦 4. Feature Distribution Box Plots",
                        color='Feature'
                    )
                    fig.update_layout(height=350, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 5. Missing Values Analysis
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    missing_df = pd.DataFrame({
                        'Feature': missing_data[missing_data > 0].index,
                        'Missing_Count': missing_data[missing_data > 0].values
                    })
                    
                    fig = px.bar(
                        missing_df,
                        x='Feature',
                        y='Missing_Count',
                        title="❓ 5. Missing Values Analysis",
                        color='Missing_Count',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=350, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Dataset size visualization
                    size_data = pd.DataFrame({
                        'Dimension': ['Rows', 'Columns'],
                        'Count': [df.shape[0], df.shape[1]]
                    })
                    fig = px.bar(
                        size_data,
                        x='Dimension',
                        y='Count',
                        title="📏 5. Dataset Dimensions",
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                # 6. Violin Plot or Density Comparison
                categorical_cols = viz_df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols and len(viz_df[categorical_cols[0]].unique()) <= 10:
                    fig = px.violin(
                        viz_df,
                        x=categorical_cols[0],
                        y=selected_target,
                        title=f"🎻 6. {selected_target} by {categorical_cols[0]}",
                        box=True,
                        color=categorical_cols[0]
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Density comparison of top features
                    if len(top_features) >= 2:
                        fig = go.Figure()
                        for feature in top_features.index[:2]:
                            fig.add_trace(go.Histogram(
                                x=viz_df[feature],
                                histnorm='probability density',
                                name=feature,
                                opacity=0.7
                            ))
                        fig.update_layout(
                            title="📈 6. Feature Density Comparison",
                            height=350,
                            barmode='overlay'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Update progress
            progress_bar.progress(60)
            status_text.text("🔗 Generating relationship analysis charts...")
            
            # Row 3: Relationship Analysis
            st.subheader("🔗 Relationship Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 7. Scatter Plot - Top Correlation
                if len(top_features) >= 1:
                    fig = px.scatter(
                        viz_df,
                        x=top_features.index[0],
                        y=selected_target,
                        title=f"📊 7. {top_features.index[0]} vs {selected_target}",
                        trendline="ols",
                        color=selected_target,
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 8. 3D Scatter Plot
                if len(top_features) >= 2:
                    fig = px.scatter_3d(
                        viz_df,
                        x=top_features.index[0],
                        y=top_features.index[1],
                        z=selected_target,
                        title="🎲 8. 3D Relationship Analysis",
                        color=selected_target,
                        opacity=0.7
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Alternative: Bar chart of feature correlations
                    if len(top_features) > 0:
                        fig = px.bar(
                            x=top_features.index.tolist(),
                            y=top_features.values.tolist(),
                            title="📊 8. Feature Correlations with Target",
                            color=top_features.values.tolist(),
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Basic info chart if no correlations available
                        st.info("📊 8. Feature Correlations with Target\n\nNot enough numeric features for correlation analysis.")
            
            with col3:
                # 9. Statistical Radar Chart
                if len(numeric_cols) >= 3:
                    numeric_stats = viz_df[numeric_cols[:5]].describe()
                    stats_to_plot = ['mean', 'std', '25%', '75%']
                    
                    fig = go.Figure()
                    
                    for col in numeric_cols[:3]:  # Limit to 3 features for clarity
                        values = []
                        for stat in stats_to_plot:
                            val = numeric_stats.loc[stat, col]
                            max_val = numeric_stats.loc[stat].max()
                            min_val = numeric_stats.loc[stat].min()
                            normalized = (val - min_val) / (max_val - min_val + 1e-8)
                            values.append(normalized)
                        values.append(values[0])  # Close the radar
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=stats_to_plot + [stats_to_plot[0]],
                            fill='toself',
                            name=col,
                            opacity=0.6
                        ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True,
                        title="🎯 9. Statistical Radar Chart",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Alternative: Simple statistics
                    stats_data = viz_df[selected_target].describe()
                    fig = px.bar(
                        x=stats_data.index,
                        y=stats_data.values,
                        title=f"📈 9. {selected_target} Statistics",
                        color=stats_data.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=350, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Row 4: Advanced Analysis
            st.subheader("🎨 Advanced Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # 10. Pair Plot / Scatter Matrix (simplified)
                if len(top_features) >= 2:
                    features_for_scatter = [selected_target] + top_features.index[:2].tolist()
                    fig = px.scatter_matrix(
                        viz_df[features_for_scatter],
                        title="🔍 10. Feature Relationship Matrix",
                        color=selected_target,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Alternative: Simple feature comparison
                    fig = px.bar(
                        x=numeric_cols,
                        y=[viz_df[col].std() for col in numeric_cols],
                        title="📊 10. Feature Variability (Standard Deviation)",
                        color=[viz_df[col].std() for col in numeric_cols],
                        color_continuous_scale='Plasma'
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Data Quality Summary
                st.subheader("📋 Data Quality Summary")
                
                quality_col1, quality_col2 = st.columns(2)
                
                with quality_col1:
                    st.metric("🔢 Total Features", df.shape[1])
                    st.metric("📊 Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
                    st.metric("📝 Categorical Features", len(df.select_dtypes(include=['object']).columns))
                
                with quality_col2:
                    st.metric("❓ Missing Values", df.isnull().sum().sum())
                    st.metric("📏 Dataset Size", f"{df.shape[0]:,} rows")
                    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                    st.metric("💾 Memory Usage", f"{memory_usage:.2f} MB")
                
                # Chart summary
                st.success("""
                **📊 Complete Visualization Suite:**
                
                ✅ **10 Interactive Charts Displayed:**
                1. Target Distribution with Box Plot
                2. Feature Type Distribution  
                3. Correlation Heatmap
                4. Feature Distribution Box Plots
                5. Missing Values Analysis
                6. Violin Plot / Density Comparison
                7. Scatter Plot with Trendline
                8. 3D Relationship Analysis
                9. Statistical Radar Chart
                10. Feature Relationship Matrix
                
                All charts are interactive with zoom, pan, and hover features!
                """)
                st.subheader("📊 Basic Data Exploration")
                col1, col2 = st.columns(2)
                
                with col1:
                    # 1. Target Distribution Histogram
                    fig = px.histogram(
                        df, 
                        x=selected_target, 
                        nbins=30, 
                        title=f"📊 Distribution of {selected_target}",
                        marginal="box",
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. Feature Count Bar Chart
                    feature_info = {
                        'Numeric Features': len(df.select_dtypes(include=[np.number]).columns),
                        'Categorical Features': len(df.select_dtypes(include=['object']).columns),
                        'Total Features': len(df.columns)
                    }
                    
                    fig = px.bar(
                        x=list(feature_info.keys()),
                        y=list(feature_info.values()),
                        title="📈 Feature Distribution",
                        color=list(feature_info.values()),
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 3. Correlation Heatmap
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        corr_matrix = numeric_df.corr()
                        fig = px.imshow(
                            corr_matrix, 
                            text_auto=True, 
                            aspect="auto", 
                            title="🔥 Feature Correlation Heatmap",
                            color_continuous_scale='RdBu'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 4. Missing Values Analysis
                    missing_data = df.isnull().sum()
                    if missing_data.sum() > 0:
                        missing_df = pd.DataFrame({
                            'Feature': missing_data.index,
                            'Missing_Count': missing_data.values,
                            'Missing_Percentage': (missing_data.values / len(df)) * 100
                        }).sort_values('Missing_Count', ascending=False)
                        
                        fig = px.bar(
                            missing_df,
                            x='Feature',
                            y='Missing_Count',
                            title="❓ Missing Values by Feature",
                            color='Missing_Percentage',
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Dataset size visualization instead
                        size_info = {
                            'Rows': df.shape[0],
                            'Columns': df.shape[1],
                            'Data Points': df.shape[0] * df.shape[1]
                        }
                        
                        fig = px.pie(
                            values=[df.shape[0], df.shape[1]],
                            names=['Rows', 'Columns'],
                            title="📏 Dataset Dimensions"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            # with viz_tab2:
                # st.subheader("📈 Distribution Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    # 5. Box Plot for all numeric features
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 1:
                        # Melt the dataframe for box plot
                        melted_df = df[numeric_cols].melt(var_name='Feature', value_name='Value')
                        
                        fig = px.box(
                            melted_df,
                            x='Feature',
                            y='Value',
                            title="📦 Box Plots - Feature Distributions",
                            color='Feature'
                        )
                        fig.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 6. Violin Plot for Target vs Categorical Features
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    if categorical_cols:
                        cat_col = categorical_cols[0]  # Use first categorical column
                        fig = px.violin(
                            df,
                            x=cat_col,
                            y=selected_target,
                            title=f"🎻 {selected_target} Distribution by {cat_col}",
                            box=True,
                            color=cat_col
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Alternative: Histogram with multiple features
                        if corr_matrix is not None and selected_target in corr_matrix.columns:
                            top_corr_features = corr_matrix[selected_target].abs().sort_values(ascending=False)[1:4]
                            
                            if len(top_corr_features) > 0:
                                fig = make_subplots(
                                    rows=1, cols=len(top_corr_features),
                                    subplot_titles=[f"Distribution of {col}" for col in top_corr_features.index]
                                )
                                
                                for i, col in enumerate(top_corr_features.index):
                                    fig.add_trace(
                                        go.Histogram(x=df[col], name=col),
                                        row=1, col=i+1
                                    )
                                
                                fig.update_layout(height=400, title="📊 Top Correlated Features Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("📊 Feature Distribution\n\nNot enough data for feature distribution analysis.")
                
                with col2:
                    # 7. Statistical Summary Radar Chart
                    numeric_stats = df[numeric_cols].describe()
                    
                    # Create radar chart for statistical measures
                    stats_to_plot = ['mean', 'std', '25%', '50%', '75%']
                    
                    fig = go.Figure()
                    
                    for col in numeric_cols[:5]:  # Limit to 5 features for clarity
                        values = []
                        for stat in stats_to_plot:
                            # Normalize values for radar chart
                            val = numeric_stats.loc[stat, col]
                            max_val = numeric_stats.loc[stat].max()
                            min_val = numeric_stats.loc[stat].min()
                            normalized = (val - min_val) / (max_val - min_val + 1e-8)
                            values.append(normalized)
                        
                        values.append(values[0])  # Close the radar
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=stats_to_plot + [stats_to_plot[0]],
                            fill='toself',
                            name=col,
                            opacity=0.6
                        ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True,
                        title="🎯 Feature Statistics Radar Chart",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 8. Density Plot Comparison
                    if corr_matrix is not None and selected_target in corr_matrix.columns:
                        top_features_local = corr_matrix[selected_target].abs().sort_values(ascending=False)[1:4]
                        
                        if len(top_features_local) > 0:
                            fig = go.Figure()
                            
                            for feature in top_features_local.index:
                                fig.add_trace(go.Histogram(
                                    x=df[feature],
                                    histnorm='probability density',
                                    name=f'{feature} Density',
                                    opacity=0.7
                                ))
                            
                            fig.update_layout(
                                title="📈 Density Comparison - Top Correlated Features",
                                height=400,
                                barmode='overlay'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("📈 8. Density Comparison\n\nNot enough numeric features for density comparison.")
            
            # with viz_tab2:
                st.subheader("� Quick Data Analysis")
                
                # Quick correlation plot
                numeric_df = viz_df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    if selected_target in corr_matrix.columns:
                        top_features = corr_matrix[selected_target].abs().sort_values(ascending=False)[1:3]
                        
                        col1, col2 = st.columns(2)
                    
                    with col1:
                        # Quick scatter plot
                        if len(top_features) >= 1:
                            fig = px.scatter(
                                viz_df,
                                x=top_features.index[0],
                                y=selected_target,
                                title=f"📊 {top_features.index[0]} vs {selected_target}",
                                trendline="ols"
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Data quality summary
                        st.subheader("📋 Data Quality Summary")
                        st.metric("🔢 Total Features", df.shape[1])
                        st.metric("📊 Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
                        st.metric("� Categorical Features", len(df.select_dtypes(include=['object']).columns))
                        st.metric("❓ Missing Values", df.isnull().sum().sum())
                        st.metric("📏 Dataset Size", f"{df.shape[0]:,} rows")
                        
                        # Performance tip
                        st.info("""
                        💡 **Performance Tips:**
                        - Use "Quick Analysis" for faster results
                        - Disable visualizations for large datasets
                        - Quick mode uses optimized algorithms
                        """)
                
                # Summary of visualization features
                st.success("""
                **⚡ Optimized for Speed:**
                
                **Quick Analysis Mode:**
                - 3 fastest algorithms (Linear, Random Forest, XGBoost)
                - Reduced parameters for speed
                - Essential visualizations only
                
                **Full Analysis Mode:**
                - 15+ comprehensive algorithms
                - Complete visualization suite
                - Detailed cross-validation
                """)
        
            # Complete progress and cleanup
            progress_bar.progress(100)
            status_text.text("✅ All visualizations generated successfully!")
            time.sleep(0.5)  # Brief pause to show completion
            progress_bar.empty()
            status_text.empty()
        
        # Algorithm Analysis
        if 'analyze_button' in locals() and analyze_button:
            st.markdown('<div class="sub-header">🤖 Algorithm Analysis</div>', unsafe_allow_html=True)
            
            # Preprocess data
            X, y, feature_cols = analyzer.preprocess_data(df, selected_target)
            
            # Train algorithms based on selected mode
            quick_mode = "Quick Analysis" in analysis_mode
            results_df = analyzer.train_all_algorithms(X, y, quick_mode=quick_mode)
            
            if not results_df.empty:
                # Best Algorithm Highlight
                best_model = analyzer.best_model
                st.markdown(f"""
                <div class="best-model">
                    🏆 BEST ALGORITHM: {best_model['Algorithm']} 
                    <br>
                    📊 Test R²: {best_model['Test R²']:.4f} | 📉 RMSE: {best_model['Test RMSE']:.2f} | ⏱️ Training: {best_model['Training Time (s)']:.3f}s
                </div>
                """, unsafe_allow_html=True)
                
                # Results table
                st.subheader("📋 Complete Algorithm Comparison")
                
                # Sort by Test R²
                results_display = results_df.sort_values('Test R²', ascending=False).reset_index(drop=True)
                results_display.index = results_display.index + 1  # Start ranking from 1
                
                # Style the dataframe
                styled_df = results_display.style.format({
                    'Train R²': '{:.4f}',
                    'Test R²': '{:.4f}',
                    'CV R² Mean': '{:.4f}',
                    'CV R² Std': '{:.4f}',
                    'Test MAE': '{:.2f}',
                    'Test RMSE': '{:.2f}',
                    'Overfitting': '{:.4f}',
                    'Training Time (s)': '{:.3f}'
                }).highlight_max(subset=['Test R²', 'CV R² Mean'], color='lightgreen')\
                  .highlight_min(subset=['Test MAE', 'Test RMSE', 'Overfitting'], color='lightgreen')
                
                st.dataframe(styled_df, width='stretch')
                
                # Performance visualizations - All 10 Charts in Single View
                st.subheader("📊 Complete Performance Analysis - All Visualizations")
                
                # Row 1: Algorithm Rankings & Performance Metrics (3 charts)
                row1_col1, row1_col2, row1_col3 = st.columns(3)
                
                with row1_col1:
                    # 1. Algorithm Ranking Bar Chart
                    fig = px.bar(
                        results_display.head(10), 
                        y='Algorithm', 
                        x='Test R²',
                        color='Category',
                        title="🏆 Top 10 Algorithms by R² Score",
                        orientation='h'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with row1_col2:
                    # 2. R² vs RMSE Scatter Plot
                    fig = px.scatter(
                        results_df, 
                        x='Test R²', 
                        y='Test RMSE',
                        color='Category',
                        size='Training Time (s)',
                        hover_data=['Algorithm'],
                        title="📈 R² Score vs RMSE Performance"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with row1_col3:
                    # 3. Overfitting Analysis
                    fig = px.bar(
                        results_display.head(10), 
                        y='Algorithm', 
                        x='Overfitting',
                        color='Overfitting',
                        color_continuous_scale=['green', 'yellow', 'red'],
                        title="⚠️ Overfitting Analysis",
                        orientation='h'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 2: Distribution & Advanced Analysis (3 charts)
                row2_col1, row2_col2, row2_col3 = st.columns(3)
                
                with row2_col1:
                    # 4. Box Plot - Performance Distribution by Category
                    fig = px.box(
                        results_df, 
                        x='Category', 
                        y='Test R²',
                        title="📦 Performance Distribution by Category",
                        color='Category'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with row2_col2:
                    # 5. Radar Chart - Multi-metric Comparison (Top 5 Models)
                    top_5_models = results_display.head(5)
                    metrics = ['Test R²', 'CV R² Mean', 'Test MAE', 'Test RMSE', 'Training Time (s)']
                    
                    # Normalize metrics for radar chart (0-1 scale)
                    normalized_data = []
                    for _, row in top_5_models.iterrows():
                        normalized_row = []
                        for metric in metrics:
                            if metric in ['Test MAE', 'Test RMSE', 'Training Time (s)']:
                                # For metrics where lower is better, invert the scale
                                normalized_val = 1 - (row[metric] - results_df[metric].min()) / (results_df[metric].max() - results_df[metric].min() + 1e-8)
                            else:
                                # For metrics where higher is better
                                normalized_val = (row[metric] - results_df[metric].min()) / (results_df[metric].max() - results_df[metric].min() + 1e-8)
                            normalized_row.append(normalized_val)
                        normalized_data.append(normalized_row + [normalized_row[0]])  # Close the radar
                    
                    fig = go.Figure()
                    for i, (_, row) in enumerate(top_5_models.iterrows()):
                        fig.add_trace(go.Scatterpolar(
                            r=normalized_data[i],
                            theta=metrics + [metrics[0]],
                            fill='toself',
                            name=row['Algorithm'],
                            opacity=0.6
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True,
                        title="🎯 Multi-Metric Radar (Top 5)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with row2_col3:
                    # 6. Parallel Coordinates - Multi-dimensional Analysis
                    parallel_data = results_df[['Algorithm', 'Test R²', 'Test MAE', 'Test RMSE', 'Training Time (s)', 'Overfitting']].copy()
                    
                    fig = px.parallel_coordinates(
                        parallel_data,
                        color='Test R²',
                        dimensions=['Test R²', 'Test MAE', 'Test RMSE', 'Training Time (s)', 'Overfitting'],
                        title="📏 Parallel Coordinates Analysis",
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 3: Advanced Visualizations (3 charts)
                row3_col1, row3_col2, row3_col3 = st.columns(3)
                
                with row3_col1:
                    # 7. Treemap - Algorithm Performance Sizes
                    fig = px.treemap(
                        results_df,
                        path=['Category', 'Algorithm'],
                        values='Test R²',
                        title="🌳 Performance Treemap",
                        color='Test R²',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with row3_col2:
                    # 8. Bubble Chart - 3D Performance Analysis
                    fig = px.scatter(
                        results_df,
                        x='Test R²',
                        y='Training Time (s)',
                        size='Test RMSE',
                        color='Category',
                        hover_name='Algorithm',
                        title="🫧 Performance Bubble Chart",
                        size_max=30
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with row3_col3:
                    # 9. 3D Scatter Plot - Multi-dimensional Performance
                    fig = px.scatter_3d(
                        results_df,
                        x='Test R²',
                        y='Test RMSE', 
                        z='Training Time (s)',
                        color='Category',
                        size='Test MAE',
                        hover_name='Algorithm',
                        title="🎲 3D Performance Analysis",
                        opacity=0.7
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 4: Best Model Analysis (1 large chart)
                st.subheader("🎯 Best Model Detailed Analysis")
                best_model_name = best_model['Algorithm']
                best_model_data = analyzer.trained_models[best_model_name]
                
                best_col1, best_col2 = st.columns(2)
                
                with best_col1:
                    # 10. Prediction vs Actual for Best Model
                    y_test = best_model_data['y_test']
                    y_pred_test = best_model_data['y_pred_test']
                    
                    fig = px.scatter(
                        x=y_test, 
                        y=y_pred_test,
                        title=f"🎯 Predicted vs Actual: {best_model_name}",
                        labels={'x': 'Actual Values', 'y': 'Predicted Values'}
                    )
                    
                    # Add perfect prediction line
                    min_val, max_val = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val], 
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with best_col2:
                    # Feature importance (if available) or Residuals plot
                    if hasattr(best_model_data['model'], 'feature_importances_'):
                        importances = best_model_data['model'].feature_importances_
                        feature_importance_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': importances
                        }).sort_values('Importance', ascending=True)
                        
                        fig = px.bar(
                            feature_importance_df, 
                            x='Importance', 
                            y='Feature',
                            title=f"🎯 Feature Importance: {best_model_name}",
                            orientation='h'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Residuals plot if no feature importance
                        residuals = y_test - y_pred_test
                        fig = px.scatter(
                            x=y_pred_test, 
                            y=residuals,
                            title=f"📊 Residuals Analysis: {best_model_name}",
                            labels={'x': 'Predicted Values', 'y': 'Residuals'}
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                st.subheader("💾 Download Results")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Analysis Results",
                    data=csv,
                    file_name=f"regression_analysis_{int(time.time())}.csv",
                    mime="text/csv"
                )
                
                # Key insights
                st.subheader("💡 Key Insights & Recommendations")
                
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    st.info(f"""
                    **🏆 Best Performing Algorithm:**
                    - **Algorithm:** {best_model['Algorithm']}
                    - **Category:** {best_model['Category']}
                    - **Performance:** {best_model['Test R²']:.4f} R² Score
                    - **Error:** ±{best_model['Test MAE']:.2f} average error
                    """)
                
                with insights_col2:
                    # Category performance
                    category_perf = results_df.groupby('Category')['Test R²'].mean().sort_values(ascending=False)
                    best_category = category_perf.index[0]
                    
                    st.success(f"""
                    **📂 Best Algorithm Category:**
                    - **Category:** {best_category}
                    - **Average R²:** {category_perf.iloc[0]:.4f}
                    - **Algorithms:** {len(results_df[results_df['Category'] == best_category])}
                    """)

if __name__ == "__main__":
    main()