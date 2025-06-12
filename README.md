### 📊 DataScope Analytics: Comprehensive Data Analysis & Machine Learning Platform
Welcome to DataScope Analytics, your all-in-one web application for streamlined data analysis, interactive visualization, and robust machine learning model building. This application empowers users to effortlessly upload, clean, analyze, and model their data through an intuitive, secure, and feature-rich interface.

### ✨ Key Features
DataScope Analytics is designed with a modular approach, offering a wide array of functionalities categorized into distinct sections:

### 🔐 Secure User Authentication
The application prioritizes data security and user privacy with a robust authentication system:

User Registration: Securely create new accounts with strong password policies (minimum length, uppercase, lowercase, number, special character requirements).

User Login: Authenticate existing users.

Account Lockout: Implements an account lockout mechanism after multiple failed login attempts to prevent brute-force attacks.

Password Change: Allows logged-in users to securely update their passwords.

Session Management: Maintains user sessions for a seamless experience.


### 📤 Data Upload & Preview
Easily bring your data into the platform:

Multiple File Formats: Supports uploading data from CSV, Excel (XLSX, XLS), and JSON files.

SQLite Database Connection: Connect directly to SQLite databases by providing a file path, listing available tables, and loading data from selected tables.

Automatic Datetime Detection: Proactively identifies and converts columns containing date/time strings into proper datetime objects upon upload, ensuring they are ready for time series analysis and feature extraction.

Data Overview: Provides an immediate summary of the uploaded dataset, including shape (rows, columns), total missing values, and duplicate rows.

Data Preview: Displays the first few rows of the dataset for quick inspection.

Column Information: Presents detailed information about each column, including data type, non-null count, null count, and unique values.


### 🧹 Data Cleaning & Preprocessing

Prepare your data for analysis and modeling with powerful cleaning tools:

Column Selection: Interactively select and extract specific columns to work with, allowing for focused analysis.

Missing Value Handling: Choose from various strategies to manage missing data:

      Drop rows with any missing values.

      Fill with mean, median, or mode for numerical columns.

      Forward fill (ffill) or backward fill (bfill) for sequential data.

Duplicate Removal: Easily identify and eliminate duplicate rows from your dataset.

Outlier Detection & Treatment: Identify and handle outliers using:

     IQR Method: Interquartile Range method to detect and either remove or cap outliers.

     Z-score Method: Detect and handle outliers based on their deviation from the mean.

Feature Engineering: Enhance your dataset with new features:

     One-Hot Encoding: Convert selected categorical columns into a numerical format suitable for machine learning models.

     Datetime Feature Extraction: Extract granular features (Year, Month, Day, Hour) from detected datetime columns, useful for time series analysis.


### 📈 Interactive Dashboard Builder
Create dynamic and insightful visualizations with Plotly:

Diverse Chart Types: Generate a variety of interactive charts:

Scatter Plot: Visualize relationships between two numerical variables.

Line Chart: Display trends over time or ordered categories.

Bar Chart: Compare categorical data.

Histogram: Show the distribution of a single numerical variable with adjustable bins.

Box Plot: Display the distribution and potential outliers of numerical data across categories.

Violin Plot: Similar to box plots but also shows the probability density of the data at different values.

Heatmap: Visualize correlation matrices for numerical features.

Pie Chart: Represent proportions of categorical data.

Choropleth Map: Visualize geographical data, mapping values to regions on a world map (requires location and value columns).

Customization Options: Tailor your visualizations with adjustable titles, heights, and color schemes.

Quick Insights: Provides immediate statistical insights related to the generated charts (e.g., correlation for scatter plots, mean/median/std for histograms).

Multi-Chart Dashboard: Automatically generates a dashboard with a selection of common charts for a holistic view of your data.

### 🤖 Machine Learning Model Builder
Build, train, evaluate, and deploy various machine learning models:

Problem Type Selection: Choose between Regression, Classification, and Time Series Forecasting tasks.

Target & Feature Selection: Select your dependent (target) and independent (feature) variables.

Model Algorithms:

Regression: Linear Regression, Random Forest Regressor, XGBoost Regressor.

Classification: Logistic Regression, Random Forest Classifier, XGBoost Classifier.

Time Series Forecasting: ARIMA (AutoRegressive Integrated Moving Average) model.

Data Preprocessing for Models:

Automatic Label Encoding for categorical features and target variables (if applicable).

Standard Scaling for numerical features to normalize data.

Robust handling of non-numeric values by coercing to numeric and filling NaNs.

Train-Test Split: Configure the ratio for splitting data into training and testing sets.

Hyperparameter Tuning: Perform Grid Search for Random Forest and XGBoost models to find optimal parameters.

Model Evaluation:

Regression: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared (R2).

Classification: Accuracy Score, Classification Report (Precision, Recall, F1-score), Confusion Matrix.

Time Series: In-sample RMSE, In-sample MAE, and interactive forecast plots with confidence intervals.

Model Deployment & Monitoring (Basic MLOps):

Model Saving: Save trained models (along with their scalers, label encoders, and evaluation metrics) to disk using joblib for persistence.

Model Loading: Load previously saved models, restoring their state and displaying their original training evaluation metrics.

Prediction Interface: Use the trained or loaded model to make predictions on new, user-inputted data, with appropriate transformations applied.


