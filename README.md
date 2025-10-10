# 🚀 Comprehensive Regression Suite

An advanced web application that brings together **15+ regression algorithms** in one interactive platform. Inspired by Giorgio De Simone's Multiple Linear Regression project, this enhanced version provides a complete machine learning workflow for regression analysis.

## ✨ Features

### 🤖 **15+ Regression Algorithms**
- **Linear Models**: Linear Regression, Ridge, Lasso, Elastic Net, Bayesian Ridge, Huber Regressor, SGD Regressor
- **Tree-Based Models**: Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost
- **Instance-Based Models**: K-Nearest Neighbors, Support Vector Regression
- **Neural Networks**: Multi-Layer Perceptron

### 📊 **Comprehensive Analysis**
- **Interactive Data Upload**: CSV file support with missing value handling
- **Feature Selection**: Choose target and feature variables with correlation analysis
- **Data Preprocessing**: Feature scaling and polynomial feature generation
- **Model Comparison**: Side-by-side performance comparison of all algorithms
- **Advanced Visualizations**: 
  - Correlation heatmaps
  - Prediction vs Actual scatter plots
  - Residuals analysis
  - Feature importance charts
  - Performance comparison charts

### 📈 **Performance Metrics**
- R² Score (Train & Test)
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Overfitting Detection
- Best Model Identification

### 🎯 **Interactive Features**
- Real-time algorithm selection
- Customizable train/test split
- Feature scaling options
- Polynomial feature engineering
- Results download (CSV format)

## 🚀 Quick Start

### Installation

1. **Clone or download the files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run comprehensive_regression_app.py
   ```

4. **Access the app**: Open your browser to `http://localhost:8501`

### Using the App

1. **📁 Upload Data**: Upload your CSV file
2. **🎯 Select Features**: Choose target variable and feature variables  
3. **🤖 Choose Algorithms**: Select one or more regression algorithms
4. **📊 Configure**: Set train/test split and preprocessing options
5. **🚀 Train Models**: Compare algorithm performance
6. **📈 Analyze Results**: View detailed visualizations and metrics
7. **💾 Download**: Export results as CSV

## 📋 Sample Data

A sample housing dataset is included (`sample_housing_data.csv`) with features:
- `Size_SqFt`: House size in square feet
- `Bedrooms`: Number of bedrooms
- `Bathrooms`: Number of bathrooms  
- `Age_Years`: Age of house in years
- `Location_Score`: Location desirability (1-10)
- `Has_Garage`: Garage availability (0/1)
- `Price`: House price (target variable)

## 🎨 Algorithm Categories

### Linear Models
Perfect for understanding feature relationships and baseline performance.

### Tree-Based Models  
Great for capturing non-linear patterns and feature interactions.

### Instance-Based Models
Effective for local pattern recognition and irregular decision boundaries.

### Neural Networks
Powerful for complex non-linear relationships and large datasets.

## 📊 Key Metrics Explained

- **R² Score**: Proportion of variance explained by the model (higher is better)
- **MAE**: Mean Absolute Error - average prediction error (lower is better)
- **RMSE**: Root Mean Square Error - penalizes large errors more (lower is better)
- **Overfitting**: Difference between train and test R² (lower is better)

## 🔬 Advanced Features

### Residuals Analysis
- Scatter plots of residuals vs predictions
- Helps identify model assumptions violations
- Detects heteroscedasticity and non-linear patterns

### Feature Importance
- Available for tree-based models
- Ranks features by their predictive power
- Helps with feature selection and interpretation

### Correlation Analysis
- Interactive heatmap of feature correlations
- Identifies multicollinearity issues
- Guides feature engineering decisions

## 🎯 Use Cases

- **Real Estate**: Housing price prediction
- **Finance**: Stock price forecasting
- **Healthcare**: Medical outcome prediction  
- **Marketing**: Sales forecasting
- **Manufacturing**: Quality prediction
- **Research**: Academic data analysis

## 🚀 Deployment Options

### Local Development
```bash
streamlit run comprehensive_regression_app.py
```

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy with one click

### Hugging Face Spaces
1. Create new Space on Hugging Face
2. Upload files
3. Automatic deployment

## 📦 Dependencies

- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms
- `plotly`: Interactive visualizations
- `matplotlib`: Static plotting
- `seaborn`: Statistical visualizations

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new regression algorithms
- Improve visualizations
- Enhance user interface
- Add new features
- Fix bugs

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Inspired by Giorgio De Simone's Multiple Linear Regression project
- Built with Streamlit and Scikit-learn
- Enhanced with comprehensive algorithm collection

---

**🚀 Comprehensive Regression Suite** - From Linear to Neural Networks, All Algorithms in One Place! 📊