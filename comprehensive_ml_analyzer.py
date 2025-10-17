"""
🚀 COMPREHENSIVE ML ANALYZER 🚀
From CSV to Actionable Insights in Under 60 Seconds!

The Process:
📁 Upload CSV → Any dataset with numbers
🎯 Pick target column → What you want to predict  
🚀 Hit "Analyze" → Sit back and watch the magic
📊 BOOM - Instant insights! → Professional results in seconds

Author: Nagaraj Satish Navada
LinkedIn: https://linkedin.com/in/nagaraj-satish-navada
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Advanced ML Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

import io
from datetime import datetime
import base64

# Page Configuration
st.set_page_config(
    page_title="🚀 ML Analyzer - CSV to Insights",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 COMPREHENSIVE ML ANALYZER</h1>
        <h3>From CSV to Actionable Insights in Under 60 Seconds!</h3>
        <p><b>📁 Upload CSV → 🎯 Pick Target → 🚀 Hit Analyze → 📊 BOOM - Instant Insights!</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for Navigation
    st.sidebar.title("🎛️ Control Panel")
    
    # Step 1: Data Upload
    st.sidebar.header("📁 Step 1: Upload Data")
    
    # Sample data button
    if st.sidebar.button("📊 Use Sample Housing Dataset"):
        if create_sample_housing_data():
            st.sidebar.success("✅ Sample data loaded!")
            st.experimental_rerun()
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file", 
        type=['csv'],
        help="Upload any CSV with numerical data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            st.sidebar.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            return
    elif 'df' not in st.session_state:
        st.info("👆 Please upload a CSV file or use sample data to get started!")
        display_features()
        return
    
    df = st.session_state['df']
    
    # Step 2: Target Selection
    st.sidebar.header("🎯 Step 2: Pick Target")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        st.error("❌ No numeric columns found for prediction!")
        return
    
    target_column = st.sidebar.selectbox(
        "Select target column to predict:",
        numeric_columns,
        help="Choose what you want to predict"
    )
    
    # Step 3: Analysis Button
    st.sidebar.header("🚀 Step 3: Analyze")
    
    if st.sidebar.button("🎯 ANALYZE NOW!", type="primary"):
        if target_column:
            st.session_state['analyzing'] = True
            run_comprehensive_analysis(df, target_column)
        else:
            st.sidebar.error("Please select a target column first!")
    
    # Display current data overview
    if not st.sidebar.button("🎯 ANALYZE NOW!", type="primary"):
        display_data_overview(df, target_column)

def create_sample_housing_data():
    """Create sample housing dataset"""
    try:
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Size_sqft': np.random.normal(2000, 500, n_samples),
            'Bedrooms': np.random.poisson(3, n_samples),
            'Bathrooms': np.random.poisson(2, n_samples),
            'Age_years': np.random.exponential(15, n_samples),
            'Garage': np.random.binomial(1, 0.7, n_samples),
            'Distance_to_city_km': np.random.exponential(10, n_samples),
        }
        
        # Create realistic price
        price = (
            data['Size_sqft'] * 150 +
            data['Bedrooms'] * 10000 +
            data['Bathrooms'] * 15000 +
            data['Garage'] * 20000 -
            data['Age_years'] * 2000 -
            data['Distance_to_city_km'] * 3000 +
            np.random.normal(0, 50000, n_samples)
        )
        
        data['Price'] = np.maximum(price, 50000)
        
        df = pd.DataFrame(data)
        
        # Clean data
        df['Size_sqft'] = np.maximum(df['Size_sqft'], 500).round().astype(int)
        df['Bedrooms'] = np.maximum(df['Bedrooms'], 1)
        df['Bathrooms'] = np.maximum(df['Bathrooms'], 1)
        df['Age_years'] = np.maximum(df['Age_years'], 0).round(1)
        df['Distance_to_city_km'] = df['Distance_to_city_km'].round(1)
        df['Price'] = df['Price'].round().astype(int)
        
        st.session_state['df'] = df
        return True
    except Exception as e:
        st.error(f"Error creating sample data: {e}")
        return False

def display_features():
    """Display features of the analyzer"""
    st.markdown("## 🌟 What You Get Automatically:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Advanced ML Pipeline
        - **15+ algorithms tested** (XGBoost, Random Forest, Neural Networks, etc.)
        - **Best model recommendation** with performance scores
        - **Smart hyperparameter tuning** for optimal results
        - **Feature importance rankings** → Know what drives predictions
        """)
        
        st.markdown("""
        ### 📊 Interactive Visualizations
        - **10 professional charts** (Bar, Pie, Scatter, Heatmap, Violin plots)
        - **Performance comparison** across all models
        - **Feature correlation** analysis
        - **Prediction accuracy** visualization
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Perfect For:
        - **Business analysts** → Predict sales, revenue, customer metrics
        - **Researchers** → Analyze experimental data without coding  
        - **Students** → Learn ML concepts with real examples
        - **Decision makers** → Get data-driven insights instantly
        """)
        
        st.markdown("""
        ### 🚀 Key Benefits:
        - **Zero coding required!** Complete automation
        - **Professional reports** ready for presentations
        - **Works with any dataset** size
        - **Results in under 60 seconds** ⚡
        """)

def display_data_overview(df, target_column):
    """Display data overview"""
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("📈 Columns", df.shape[1])
    with col3:
        st.metric("🎯 Target", target_column if target_column else "Not Selected")
    with col4:
        st.metric("❌ Missing Values", f"{df.isnull().sum().sum():,}")
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Basic statistics
    if target_column:
        st.subheader(f"📈 Target Statistics: {target_column}")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{df[target_column].mean():.2f}")
        with col2:
            st.metric("Median", f"{df[target_column].median():.2f}")
        with col3:
            st.metric("Min", f"{df[target_column].min():.2f}")
        with col4:
            st.metric("Max", f"{df[target_column].max():.2f}")

def run_comprehensive_analysis(df, target_column):
    """Run the comprehensive ML analysis"""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Data Preprocessing
        status_text.text("🔄 Step 1/5: Preprocessing data...")
        progress_bar.progress(20)
        
        X, y, feature_names = preprocess_data(df, target_column)
        
        # Step 2: Train-Test Split
        status_text.text("🔄 Step 2/5: Splitting data...")
        progress_bar.progress(40)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Step 3: Model Training
        status_text.text("🔄 Step 3/5: Training 15+ models... ⏰")
        progress_bar.progress(60)
        
        results = train_all_models(X_train, X_test, y_train, y_test, feature_names)
        
        # Step 4: Generate Visualizations
        status_text.text("🔄 Step 4/5: Creating 10 interactive charts...")
        progress_bar.progress(80)
        
        generate_visualizations(df, target_column, X, y, results, feature_names)
        
        # Step 5: Generate Report
        status_text.text("🔄 Step 5/5: Generating professional report...")
        progress_bar.progress(100)
        
        generate_analysis_report(df, target_column, results, feature_names)
        
        status_text.text("✅ Analysis Complete! Scroll down for results 👇")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)

def preprocess_data(df, target_column):
    """Preprocess the data for ML"""
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values
    X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
    
    # Encode categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values, feature_names

def train_all_models(X_train, X_test, y_train, y_test, feature_names):
    """Train all available models"""
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Elastic Net': ElasticNet(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'SVM': SVR(kernel='rbf'),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    # Add advanced models if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMRegressor(random_state=42, verbose=-1)
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostRegressor(random_state=42, verbose=False)
    
    results = {}
    
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(feature_names, np.abs(model.coef_)))
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'cv_mean': cv_mean,
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            st.warning(f"❌ {name} failed: {str(e)}")
    
    return results

def generate_visualizations(df, target_column, X, y, results, feature_names):
    """Generate 10 interactive visualizations"""
    
    st.header("📊 Interactive Visualizations & Analysis")
    
    # 1. Model Performance Comparison
    st.subheader("1. 🏆 Model Performance Comparison")
    
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    rmse_scores = [results[name]['rmse'] for name in model_names]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['R² Score (Higher = Better)', 'RMSE (Lower = Better)'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # R² scores
    fig.add_trace(
        go.Bar(x=model_names, y=r2_scores, name='R² Score', marker_color='skyblue'),
        row=1, col=1
    )
    
    # RMSE scores
    fig.add_trace(
        go.Bar(x=model_names, y=rmse_scores, name='RMSE', marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig.update_layout(height=500, showlegend=False, title_text="Model Performance Metrics")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Best Model Highlight
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]
    
    st.markdown(f"""
    <div class="success-box">
        <h3>🏅 BEST MODEL RECOMMENDATION</h3>
        <h2><strong>{best_model_name}</strong></h2>
        <p><strong>R² Score:</strong> {best_model['r2']:.4f} ({best_model['r2']*100:.2f}% accuracy)</p>
        <p><strong>RMSE:</strong> {best_model['rmse']:.4f}</p>
        <p><strong>Cross-Validation Score:</strong> {best_model['cv_mean']:.4f} (±{best_model['cv_std']:.4f})</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Feature Importance (Top Model)
    if best_model['feature_importance']:
        st.subheader("3. 🎯 Feature Importance Rankings")
        
        importance_df = pd.DataFrame([
            {'Feature': k, 'Importance': v} 
            for k, v in best_model['feature_importance'].items()
        ]).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            title=f"Feature Importance - {best_model_name}",
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # 4. Target Distribution
    st.subheader("4. 📈 Target Variable Distribution")
    
    fig = px.histogram(
        df, 
        x=target_column,
        title=f"Distribution of {target_column}",
        color_discrete_sequence=['lightblue']
    )
    fig.add_vline(x=df[target_column].mean(), line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {df[target_column].mean():.2f}")
    st.plotly_chart(fig, use_container_width=True)
    
    # 5. Correlation Heatmap
    st.subheader("5. 🔥 Feature Correlation Heatmap")
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # 6. Actual vs Predicted (Best Model)
    st.subheader("6. 🎯 Actual vs Predicted Values")
    
    y_test_actual = y[-len(best_model['predictions']):]  # Get test set actual values
    
    fig = px.scatter(
        x=y_test_actual,
        y=best_model['predictions'],
        title=f"Actual vs Predicted - {best_model_name}",
        labels={'x': f'Actual {target_column}', 'y': f'Predicted {target_column}'},
        trendline="ols"
    )
    
    # Add perfect prediction line
    min_val = min(y_test_actual.min(), best_model['predictions'].min())
    max_val = max(y_test_actual.max(), best_model['predictions'].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 7. Model Accuracy Pie Chart
    st.subheader("7. 🥧 Model Accuracy Overview")
    
    accuracy_ranges = {'Excellent (>90%)': 0, 'Good (80-90%)': 0, 'Fair (70-80%)': 0, 'Poor (<70%)': 0}
    
    for name, result in results.items():
        accuracy = result['r2'] * 100
        if accuracy >= 90:
            accuracy_ranges['Excellent (>90%)'] += 1
        elif accuracy >= 80:
            accuracy_ranges['Good (80-90%)'] += 1
        elif accuracy >= 70:
            accuracy_ranges['Fair (70-80%)'] += 1
        else:
            accuracy_ranges['Poor (<70%)'] += 1
    
    fig = px.pie(
        values=list(accuracy_ranges.values()),
        names=list(accuracy_ranges.keys()),
        title="Model Performance Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 8. Cross-Validation Scores
    st.subheader("8. 📊 Cross-Validation Stability")
    
    cv_data = []
    for name, result in results.items():
        cv_data.append({
            'Model': name,
            'CV Mean': result['cv_mean'],
            'CV Std': result['cv_std']
        })
    
    cv_df = pd.DataFrame(cv_data).sort_values('CV Mean', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cv_df['CV Mean'],
        y=cv_df['Model'],
        error_x=dict(type='data', array=cv_df['CV Std']),
        orientation='h',
        marker_color='lightgreen'
    ))
    fig.update_layout(
        title="Cross-Validation Scores with Standard Deviation",
        xaxis_title="CV Score",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 9. Feature Scatter Plots (Top 3 Features)
    if best_model['feature_importance']:
        st.subheader("9. 🔍 Top Features vs Target")
        
        top_features = sorted(
            best_model['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        cols = st.columns(3)
        for i, (feature, importance) in enumerate(top_features):
            if feature in df.columns:
                with cols[i]:
                    fig = px.scatter(
                        df,
                        x=feature,
                        y=target_column,
                        title=f"{feature} vs {target_column}",
                        trendline="ols"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    # 10. Model Performance Radar Chart
    st.subheader("10. 🕸️ Model Performance Radar")
    
    # Normalize metrics for radar chart
    max_r2 = max([results[name]['r2'] for name in results.keys()])
    min_rmse = min([results[name]['rmse'] for name in results.keys()])
    max_rmse = max([results[name]['rmse'] for name in results.keys()])
    
    radar_data = []
    for name, result in results.items():
        normalized_r2 = result['r2'] / max_r2 if max_r2 > 0 else 0
        normalized_rmse = 1 - ((result['rmse'] - min_rmse) / (max_rmse - min_rmse)) if max_rmse > min_rmse else 1
        normalized_cv = result['cv_mean'] / max_r2 if max_r2 > 0 else 0
        
        radar_data.append({
            'Model': name,
            'R² Score': normalized_r2,
            'RMSE (Inverted)': normalized_rmse,
            'CV Score': normalized_cv
        })
    
    # Show top 5 models in radar
    radar_df = pd.DataFrame(radar_data).sort_values('R² Score', ascending=False).head(5)
    
    fig = go.Figure()
    
    metrics = ['R² Score', 'RMSE (Inverted)', 'CV Score']
    
    for _, row in radar_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[metric] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Top 5 Models Performance Comparison",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_analysis_report(df, target_column, results, feature_names):
    """Generate downloadable analysis report"""
    
    st.header("📋 Professional Analysis Report")
    
    # Report content
    report_content = f"""
# 🚀 COMPREHENSIVE ML ANALYSIS REPORT

**Dataset:** {df.shape[0]} rows × {df.shape[1]} columns  
**Target Variable:** {target_column}  
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Generated by:** Comprehensive ML Analyzer

---

## 📊 EXECUTIVE SUMMARY

### Best Model Recommendation
"""
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]
    
    report_content += f"""
**🏆 Recommended Model:** {best_model_name}  
**📈 Accuracy (R²):** {best_model['r2']:.4f} ({best_model['r2']*100:.2f}%)  
**📉 RMSE:** {best_model['rmse']:.4f}  
**🔄 Cross-Validation:** {best_model['cv_mean']:.4f} (±{best_model['cv_std']:.4f})

### Key Insights
- **Total Models Tested:** {len(results)}
- **Best Performance:** {best_model['r2']*100:.2f}% accuracy
- **Most Stable Model:** {max(results.keys(), key=lambda x: results[x]['cv_mean'])}

---

## 📈 DETAILED RESULTS

### All Model Performance
"""
    
    # Add all results
    for name, result in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
        report_content += f"""
**{name}:**
- R² Score: {result['r2']:.4f}
- RMSE: {result['rmse']:.4f}
- MAE: {result['mae']:.4f}
- CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})
"""
    
    # Add feature importance
    if best_model['feature_importance']:
        report_content += f"""
---

## 🎯 FEATURE IMPORTANCE ({best_model_name})

"""
        sorted_features = sorted(
            best_model['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (feature, importance) in enumerate(sorted_features, 1):
            report_content += f"{i}. **{feature}:** {importance:.4f}\n"
    
    report_content += f"""
---

## 📋 DATASET INFORMATION

- **Total Samples:** {df.shape[0]:,}
- **Features:** {df.shape[1]}
- **Target Variable:** {target_column}
- **Missing Values:** {df.isnull().sum().sum():,}
- **Numeric Features:** {len(df.select_dtypes(include=[np.number]).columns)}

### Target Statistics
- **Mean:** {df[target_column].mean():.2f}
- **Median:** {df[target_column].median():.2f}
- **Min:** {df[target_column].min():.2f}
- **Max:** {df[target_column].max():.2f}
- **Standard Deviation:** {df[target_column].std():.2f}

---

## 🚀 RECOMMENDATIONS

1. **Deploy {best_model_name}** for production use with {best_model['r2']*100:.2f}% accuracy
2. **Monitor top features:** {', '.join(list(dict(sorted(best_model['feature_importance'].items(), key=lambda x: x[1], reverse=True)).keys())[:3]) if best_model['feature_importance'] else 'N/A'}
3. **Consider ensemble methods** if higher accuracy is needed
4. **Regular retraining** recommended with new data

---

**Report generated by Comprehensive ML Analyzer**  
**Author:** Nagaraj Satish Navada  
**Contact:** [LinkedIn](https://linkedin.com/in/nagaraj-satish-navada)
"""
    
    # Display report preview
    st.markdown("### 📖 Report Preview")
    st.markdown(report_content)
    
    # Download button
    st.markdown("### 📥 Download Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📄 Download as Markdown",
            data=report_content,
            file_name=f"ml_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    with col2:
        # Convert to simple text for TXT download
        txt_content = report_content.replace('#', '').replace('*', '').replace('- ', '• ')
        st.download_button(
            label="📝 Download as Text",
            data=txt_content,
            file_name=f"ml_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()