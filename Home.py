import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score, mean_absolute_error # Added MAE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats # For T-test
from sklearn.cluster import KMeans # New import for K-Means
from sklearn.manifold import TSNE # New import for t-SNE
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import hashlib
import sqlite3
import re
from datetime import datetime, timedelta
import secrets
import joblib # New import for saving/loading models
import requests # New import for API calls
import json # New import for JSON handling

# New import for Time Series Forecasting
from statsmodels.tsa.arima.model import ARIMA
# Suppress specific warnings from statsmodels
warnings.filterwarnings("ignore", message="No supported index is available. Prediction results will be given with default index", category=UserWarning)
warnings.warn("A date index must be provided for the predict method", UserWarning) # Corrected syntax for warnings.warn


# Ensure the directory for saved models exists
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Database setup
def init_database():
    """Initialize SQLite database for user management"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            failed_attempts INTEGER DEFAULT 0,
            locked_until TIMESTAMP
        )
    ''')
    
    # Create password reset tokens table (for future email functionality)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            token TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            used BOOLEAN DEFAULT FALSE
        )
    ''')
    
    conn.commit()
    conn.close()

# Password utilities
def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed_password):
    """Verify password against hash"""
    return hash_password(password) == hashed_password

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, "Password is valid"

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# User management functions
def create_user(username, email, password):
    """Create a new user"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        ''', (username, email, password_hash))
        conn.commit()
        return True, "User created successfully"
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already exists"
        elif "email" in str(e):
            return False, "Email already exists"
        else:
            return False, "User creation failed"
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticate user credentials"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        # Check if account is locked
        cursor.execute('''
            SELECT locked_until FROM users 
            WHERE username = ? OR email = ?
        ''', (username, username))
        
        result = cursor.fetchone()
        if result and result[0]:
            locked_until = datetime.fromisoformat(result[0])
            if datetime.now() < locked_until:
                return False, "Account is temporarily locked due to too many failed attempts"
        
        # Verify credentials
        cursor.execute('''
            SELECT id, username, email, password_hash, failed_attempts 
            FROM users 
            WHERE (username = ? OR email = ?) AND is_active = TRUE
        ''', (username, username))
        
        user = cursor.fetchone()
        if user and verify_password(password, user[3]):
            # Reset failed attempts and update last login
            cursor.execute('''
                UPDATE users 
                SET failed_attempts = 0, locked_until = NULL, last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (user[0],))
            conn.commit()
            return True, {"id": user[0], "username": user[1], "email": user[2]}
        else:
            # Increment failed attempts
            if user:
                failed_attempts = user[4] + 1
                locked_until = None
                if failed_attempts >= 5:
                    locked_until = (datetime.now() + timedelta(minutes=15)).isoformat()
                
                cursor.execute('''
                    UPDATE users 
                    SET failed_attempts = ?, locked_until = ?
                    WHERE id = ?
                ''', (failed_attempts, locked_until, user[0]))
                conn.commit()
            
            return False, "Invalid username/email or password"
    
    except Exception as e:
        return False, f"Authentication error: {str(e)}"
    finally:
        conn.close()

def change_password(user_id, old_password, new_password):
    """Change user password"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        # Verify old password
        cursor.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,))
        result = cursor.fetchone()
        
        if not result or not verify_password(old_password, result[0]):
            return False, "Current password is incorrect"
        
        # Validate new password
        is_valid, message = validate_password(new_password)
        if not is_valid:
            return False, message
        
        # Update password
        new_hash = hash_password(new_password)
        cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user_id))
        conn.commit()
        return True, "Password changed successfully"
    
    except Exception as e:
        return False, f"Error changing password: {str(e)}"
    finally:
        conn.close()

# Session management
def init_session_state():
    """Initialize session state variables"""
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'email' not in st.session_state:
        st.session_state.email = None
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0
    # Initialize DataScope app specific session states
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'show_profile' not in st.session_state:
        st.session_state.show_profile = False
    if 'trained_model' not in st.session_state: # For prediction interface
        st.session_state.trained_model = None
    if 'model_features' not in st.session_state: # For prediction interface
        st.session_state.model_features = None
    if 'model_target' not in st.session_state: # For prediction interface
        st.session_state.model_target = None
    if 'model_problem_type' not in st.session_state: # For prediction interface
        st.session_state.model_problem_type = None
    if 'model_scaler' not in st.session_state: # For prediction interface
        st.session_state.model_scaler = None
    if 'model_label_encoders' not in st.session_state: # For prediction interface
        st.session_state.model_label_encoders = {}
    if 'model_evaluation_metrics' not in st.session_state: # For displaying loaded model metrics
        st.session_state.model_evaluation_metrics = {}


def logout():
    """Clear session state and logout user"""
    # Clear all session state variables related to the user and app data
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Re-initialize only authentication and essential session states
    init_session_state()
    st.rerun()

# UI Components
def login_form():
    """Display login form"""
    st.header("🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username or Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username and password:
                success, result = authenticate_user(username, password)
                if success:
                    st.session_state.authentication_status = True
                    st.session_state.username = result["username"]
                    st.session_state.user_id = result["id"]
                    st.session_state.email = result["email"]
                    st.session_state.login_attempts = 0
                    st.success(f"Welcome back, {result['username']}!")
                    st.rerun()
                else:
                    st.session_state.authentication_status = False
                    st.session_state.login_attempts += 1
                    st.error(result)
            else:
                st.error("Please enter both username/email and password")

def registration_form():
    """Display registration form"""
    st.header("📝 Create Account")
    
    with st.form("registration_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account")
        
        if submitted:
            if not all([username, email, password, confirm_password]):
                st.error("Please fill in all fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif not validate_email(email):
                st.error("Please enter a valid email address")
            else:
                is_valid, message = validate_password(password)
                if not is_valid:
                    st.error(message)
                else:
                    success, result = create_user(username, email, password)
                    if success:
                        st.success("Account created successfully! You can now log in.")
                        st.balloons()
                    else:
                        st.error(result)

def password_change_form():
    """Display password change form"""
    st.header("🔑 Change Password")
    
    with st.form("password_change_form"):
        old_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        submitted = st.form_submit_button("Change Password")
        
        if submitted:
            if not all([old_password, new_password, confirm_password]):
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("New passwords do not match")
            else:
                success, message = change_password(st.session_state.user_id, old_password, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

def user_profile():
    """Display user profile information"""
    st.header("👤 User Profile")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Username:** {st.session_state.username}")
        st.info(f"**Email:** {st.session_state.email}")
    
    with col2:
        if st.button("🚪 Logout", type="primary"):
            logout()
    
    st.divider()
    password_change_form()

# Helper functions for the DataScope app
def get_download_link(df, filename="data.csv"):
    """Generate download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

def detect_column_types(df):
    """Automatically detect column types"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = []
    
    # Iterate through all columns, not just categorical, to catch already converted datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_object_dtype(df[col]):
            # Attempt to convert to datetime for object columns
            temp_series = pd.to_datetime(df[col], errors='coerce')
            # If more than 50% non-null datetime values, consider it a datetime column
            if temp_series.count() / len(temp_series) > 0.5: 
                datetime_cols.append(col)
                # Note: The actual conversion in the dataframe should happen earlier,
                # this function just detects based on current dtypes or potential for conversion.
    
    # Filter out columns that are now datetime from categorical
    categorical_cols = [col for col in categorical_cols if col not in datetime_cols]
    
    return numeric_cols, categorical_cols, datetime_cols


def datascope_app_content():
    """All the content for the DataScope Analytics application"""
    
    # Main title
    st.markdown('<h1 class="main-header">📊 DataScope Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### *Comprehensive Data Analytics & Dashboard Builder*")

    # Sidebar navigation for the DataScope app modules
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.selectbox(
        "Choose a module:",
        ["🏠 Home", "📤 Data Upload", "🧹 Data Cleaning", "📈 Dashboard Builder", "🤖 Model Builder", "📊 Advanced Analytics", "📈 Business Intelligence", "💰 Financial Analytics", "✨ Enhanced Analytics & Visualization"] # Added Enhanced Analytics
    )

    # HOME PAGE
    if page == "🏠 Home":
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ## Welcome to DataScope Analytics! 🎯
            
            Your all-in-one platform for data analytics, visualization, and machine learning.
            
            ### 🌟 Key Features:
            """)
            
            features = [
                "📤 **Data Upload**: Support for CSV, Excel, JSON, SQLite, and **API data fetching**", # Updated feature list
                "🧹 **Data Cleaning**: Handle missing values, duplicates, outliers, and advanced feature engineering",
                "📈 **Dashboard Builder**: Create interactive visualizations with Plotly, including new chart types like Choropleth Maps and Violin Plots",
                "🤖 **Model Builder**: Build and evaluate ML models (Linear, Logistic, Random Forest, XGBoost) with hyperparameter tuning and a prediction interface",
                "📊 **Advanced Analytics**: Statistical analysis, PCA, and deeper insights",
                "📈 **Business Intelligence**: Advanced reporting, executive dashboards, and interactive storytelling.",
                "💰 **Financial Analytics**: Risk modeling, portfolio optimization, and financial metrics.",
                "✨ **Enhanced Analytics & Visualization**: Advanced statistical methods and interactive visualization capabilities."
            ]
            
            for feature in features:
                st.markdown(f"- {feature}")
            
            st.markdown("---")
            st.info("👈 Use the sidebar to navigate through different modules and start your data journey!")

            # Automated Basic Insights
            if st.session_state.data is not None:
                st.markdown("### 💡 Quick Data Insights")
                df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
                numeric_cols, categorical_cols, _ = detect_column_types(df)

                if not df.empty:
                    st.write(f"Dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
                    st.write(f"Total missing values: **{df.isnull().sum().sum()}**.")
                    st.write(f"Number of duplicate rows: **{df.duplicated().sum()}**.")

                    if numeric_cols:
                        st.subheader("Numerical Insights:")
                        for col in numeric_cols[:3]: # Show for first 3 numeric columns
                            st.write(f"- **{col}**: Mean = {df[col].mean():.2f}, Median = {df[col].median():.2f}, Std Dev = {df[col].std():.2f}")
                    
                    if categorical_cols:
                        st.subheader("Categorical Insights:")
                        for col in categorical_cols[:3]: # Show for first 3 categorical columns
                            top_category = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                            st.write(f"- **{col}**: Most frequent category is '{top_category}' (Count: {df[col].value_counts().max()})")
                else:
                    st.info("Upload data to see quick insights!")


    # DATA UPLOAD PAGE
    elif page == "📤 Data Upload":
        st.markdown('<h2 class="section-header">📤 Data Upload & Preview</h2>', unsafe_allow_html=True)
        
        tab_files, tab_db, tab_api = st.tabs(["Upload Files", "Connect to SQLite Database", "Fetch from API"])

        with tab_files:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a file to upload",
                type=['csv', 'xlsx', 'xls', 'json'],
                help="Supported formats: CSV, Excel, JSON"
            )
            
            if uploaded_file is not None:
                try:
                    # Read file based on type
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        df = pd.read_json(uploaded_file)
                    
                    # Proactively attempt to convert object columns to datetime
                    for col in df.select_dtypes(include='object').columns:
                        try:
                            # Attempt to convert to datetime, coercing errors to NaT
                            temp_series = pd.to_datetime(df[col], errors='coerce')
                            # If a significant portion (e.g., > 50%) of values successfully converted,
                            # or if the column was entirely non-null and now has some NaT,
                            # consider it a datetime column and update its dtype in the DataFrame.
                            # The threshold helps avoid converting columns that are mostly text but have a few date-like strings.
                            if temp_series.count() > 0 and (temp_series.count() / len(temp_series) > 0.5 or temp_series.isnull().sum() == 0):
                                df[col] = temp_series
                                st.info(f"Column '{col}' successfully converted to datetime.")
                        except Exception:
                            # Ignore columns that cannot be converted to datetime
                            pass

                    st.session_state.data = df
                    st.session_state.cleaned_data = df.copy()
                    
                    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                    
                    # Display basic info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", df.shape[0])
                    with col2:
                        st.metric("Columns", df.shape[1])
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    with col4:
                        st.metric("Duplicates", df.duplicated().sum())
                    
                    # Preview data
                    st.markdown("### 👀 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Column information
                    st.markdown("### 📋 Column Information")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Null Count': df.isnull().sum(),
                        'Unique Values': df.nunique()
                    })
                    st.dataframe(col_info, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
            
            else:
                st.info("📁 Please upload a file to get started!")

        with tab_db:
            st.markdown("### Connect to SQLite Database")
            db_path = st.text_input("Enter SQLite database file path (e.g., my_database.db):", "users.db")
            
            if st.button("Connect to Database"):
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [table[0] for table in cursor.fetchall()]
                    conn.close()
                    
                    if tables:
                        st.success(f"Connected to '{db_path}'. Found tables: {', '.join(tables)}")
                        st.session_state.db_tables = tables
                        st.session_state.db_path = db_path
                    else:
                        st.warning(f"No tables found in '{db_path}'.")
                        st.session_state.db_tables = []
                except Exception as e:
                    st.error(f"❌ Error connecting to database: {str(e)}")
            
            if 'db_tables' in st.session_state and st.session_state.db_tables:
                selected_table = st.selectbox("Select a table to load:", st.session_state.db_tables)
                if st.button(f"Load Data from {selected_table}"):
                    try:
                        conn = sqlite3.connect(st.session_state.db_path)
                        df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                        conn.close()

                        # Proactively attempt to convert object columns to datetime for DB data too
                        for col in df.select_dtypes(include='object').columns:
                            try:
                                temp_series = pd.to_datetime(df[col], errors='coerce')
                                if temp_series.count() > 0 and (temp_series.count() / len(temp_series) > 0.5 or temp_series.isnull().sum() == 0):
                                    df[col] = temp_series
                                    st.info(f"Column '{col}' successfully converted to datetime.")
                            except Exception:
                                pass

                        st.session_state.data = df
                        st.session_state.cleaned_data = df.copy()
                        st.success(f"✅ Data loaded from table '{selected_table}' successfully! Shape: {df.shape}")
                        st.dataframe(df.head(10), use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error loading data from table: {str(e)}")
        
        with tab_api:
            st.markdown("### Fetch Data from API (JSON)")
            api_url = st.text_input("Enter API URL (e.g., https://api.exchangerate-api.com/v4/latest/USD)")
            api_headers_str = st.text_area("Optional: Enter Headers as JSON (e.g., {'Authorization': 'Bearer YOUR_TOKEN'})", "{}")
            
            if st.button("Fetch Data from API"):
                if api_url:
                    try:
                        headers = json.loads(api_headers_str) if api_headers_str else {}
                        
                        st.info(f"Fetching data from {api_url}...")
                        response = requests.get(api_url, headers=headers)
                        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                        
                        json_data = response.json()
                        
                        # Attempt to flatten JSON data to DataFrame
                        try:
                            # Try to normalize if it's a list of dicts or nested dicts
                            df = pd.json_normalize(json_data)
                        except Exception as e:
                            st.warning(f"Could not automatically flatten JSON. Attempting direct DataFrame conversion. Error: {e}")
                            # If json_normalize fails, try direct conversion (e.g., if it's a flat dict)
                            df = pd.DataFrame([json_data])
                        
                        if df.empty:
                            st.warning("API returned empty data or could not be converted to DataFrame.")
                            st.stop()

                        # Proactively attempt to convert object columns to datetime
                        for col in df.select_dtypes(include='object').columns:
                            try:
                                temp_series = pd.to_datetime(df[col], errors='coerce')
                                if temp_series.count() > 0 and (temp_series.count() / len(temp_series) > 0.5 or temp_series.isnull().sum() == 0):
                                    df[col] = temp_series
                                    st.info(f"Column '{col}' successfully converted to datetime.")
                            except Exception:
                                pass

                        st.session_state.data = df
                        st.session_state.cleaned_data = df.copy()
                        
                        st.success(f"✅ Data fetched from API successfully! Shape: {df.shape}")
                        
                        # Display basic info
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Rows", df.shape[0])
                        with col2:
                            st.metric("Columns", df.shape[1])
                        with col3:
                            st.metric("Missing Values", df.isnull().sum().sum())
                        with col4:
                            st.metric("Duplicates", df.duplicated().sum())
                        
                        # Preview data
                        st.markdown("### 👀 Data Preview")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Column information
                        st.markdown("### 📋 Column Information")
                        col_info = pd.DataFrame({
                            'Column': df.columns,
                            'Data Type': df.dtypes,
                            'Non-Null Count': df.count(),
                            'Null Count': df.isnull().sum(),
                            'Unique Values': df.nunique()
                        })
                        st.dataframe(col_info, use_container_width=True)

                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Network or API request error: {str(e)}")
                    except json.JSONDecodeError:
                        st.error("❌ API response is not a valid JSON. Please check the URL.")
                    except Exception as e:
                        st.error(f"❌ An unexpected error occurred: {str(e)}")
                else:
                    st.warning("Please enter an API URL.")


    # DATA CLEANING PAGE
    elif page == "🧹 Data Cleaning":
        st.markdown('<h2 class="section-header">🧹 Data Cleaning & Preprocessing</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.cleaned_data.copy()
            
            # Column Selection & Extraction Section
            st.markdown("### ✂️ Column Selection & Extraction")
            all_current_cols = df.columns.tolist()
            if all_current_cols:
                columns_to_keep = st.multiselect(
                    "Select columns to keep in your dataset:",
                    all_current_cols,
                    default=all_current_cols # Default to keeping all columns
                )
                
                if st.button("Extract Selected Columns"):
                    if columns_to_keep:
                        try:
                            extracted_df = df[columns_to_keep].copy()
                            st.session_state.cleaned_data = extracted_df
                            st.success(f"✅ Columns extracted successfully! New shape: {extracted_df.shape}")
                            st.dataframe(extracted_df.head(10), use_container_width=True)
                        except Exception as e:
                            st.error(f"❌ Error extracting columns: {str(e)}")
                    else:
                        st.warning("Please select at least one column to extract.")
            else:
                st.info("No columns available to select. Please upload data first.")

            st.markdown("---") # Separator for clarity

            # Cleaning options
            st.markdown("### 🔧 Cleaning Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Missing Values")
                missing_action = st.selectbox(
                    "Handle missing values:",
                    ["No action", "Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with mode", "Forward fill", "Backward fill"]
                )
                
                st.markdown("#### Duplicates")
                remove_duplicates = st.checkbox("Remove duplicate rows")
                
            with col2:
                st.markdown("#### Outliers")
                outlier_method = st.selectbox(
                    "Outlier detection method:",
                    ["No action", "IQR method", "Z-score method"]
                )
                
                if outlier_method != "No action":
                    outlier_action = st.selectbox(
                        "Action for outliers:",
                        ["Remove", "Cap values"]
                    )

            st.markdown("### ✨ Feature Engineering")
            numeric_cols, categorical_cols, datetime_cols = detect_column_types(df) # Recalculate based on potentially extracted df

            # One-Hot Encoding
            if categorical_cols:
                ohe_cols = st.multiselect("Select categorical columns for One-Hot Encoding:", categorical_cols)
            else:
                ohe_cols = []
                st.info("No categorical columns found for One-Hot Encoding.")

            # Date/Time Feature Extraction
            if datetime_cols:
                st.markdown("#### Date/Time Feature Extraction")
                selected_dt_col = st.selectbox("Select a datetime column to extract features from:", ["None"] + datetime_cols)
                if selected_dt_col != "None":
                    extract_year = st.checkbox("Extract Year")
                    extract_month = st.checkbox("Extract Month")
                    extract_day = st.checkbox("Extract Day")
                    extract_hour = st.checkbox("Extract Hour")
            else:
                selected_dt_col = "None"
                st.info("No datetime columns found for feature extraction.")

            
            # Apply cleaning
            if st.button("🔄 Apply Cleaning & Feature Engineering"):
                cleaned_df = df.copy() # Start with the current cleaned_data (which might be extracted)
                
                # Handle missing values
                if missing_action == "Drop rows with missing values":
                    cleaned_df = cleaned_df.dropna()
                elif missing_action == "Fill with mean":
                    numeric_cols_df = cleaned_df.select_dtypes(include=[np.number]).columns
                    cleaned_df[numeric_cols_df] = cleaned_df[numeric_cols_df].fillna(cleaned_df[numeric_cols_df].mean())
                elif missing_action == "Fill with median":
                    numeric_cols_df = cleaned_df.select_dtypes(include=[np.number]).columns
                    cleaned_df[numeric_cols_df] = cleaned_df[numeric_cols_df].fillna(cleaned_df[numeric_cols_df].median())
                elif missing_action == "Fill with mode":
                    for col in cleaned_df.columns:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else cleaned_df[col])
                elif missing_action == "Forward fill":
                    cleaned_df = cleaned_df.fillna(method='ffill')
                elif missing_action == "Backward fill":
                    cleaned_df = cleaned_df.fillna(method='bfill')
                
                # Remove duplicates
                if remove_duplicates:
                    cleaned_df = cleaned_df.drop_duplicates()
                
                # Handle outliers
                if outlier_method != "No action":
                    numeric_cols_df = cleaned_df.select_dtypes(include=[np.number]).columns
                    
                    for col in numeric_cols_df:
                        # Ensure column is numeric and not empty before calculating quantiles
                        if pd.api.types.is_numeric_dtype(cleaned_df[col]) and not cleaned_df[col].empty:
                            Q1 = cleaned_df[col].quantile(0.25)
                            Q3 = cleaned_df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            
                            if outlier_method == "IQR method":
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                outliers = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
                            else:  # Z-score method
                                # Calculate z-scores only for non-null values
                                valid_data = cleaned_df[col].dropna()
                                if not valid_data.empty and valid_data.std() != 0:
                                    z_scores = np.abs((valid_data - valid_data.mean()) / valid_data.std())
                                    outliers = z_scores > 3
                                    # Map back to original index
                                    outliers = cleaned_df[col].index.isin(outliers[outliers].index)
                                else:
                                    outliers = pd.Series(False, index=cleaned_df.index) # No outliers if no valid data or std is 0
                            
                            if outliers.any():
                                if outlier_action == "Remove":
                                    cleaned_df = cleaned_df[~outliers]
                                else:  # Cap values
                                    cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
                                    cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
            
                # Apply One-Hot Encoding
                if ohe_cols:
                    try:
                        cleaned_df = pd.get_dummies(cleaned_df, columns=ohe_cols, drop_first=True) # drop_first to avoid multicollinearity
                        st.success(f"One-Hot Encoding applied to: {', '.join(ohe_cols)}")
                    except Exception as e:
                        st.error(f"Error applying One-Hot Encoding: {e}")

                # Apply Date/Time Feature Extraction
                if selected_dt_col != "None" and selected_dt_col in cleaned_df.columns:
                    # Ensure the column is indeed datetime type before extracting
                    cleaned_df[selected_dt_col] = pd.to_datetime(cleaned_df[selected_dt_col], errors='coerce')
                    if cleaned_df[selected_dt_col].isnull().all():
                        st.warning(f"Cannot extract date/time features from '{selected_dt_col}' as it contains no valid datetime values after coercion.")
                    else:
                        if extract_year:
                            cleaned_df[f'{selected_dt_col}_Year'] = cleaned_df[selected_dt_col].dt.year
                        if extract_month:
                            cleaned_df[f'{selected_dt_col}_Month'] = cleaned_df[selected_dt_col].dt.month
                        if extract_day:
                            cleaned_df[f'{selected_dt_col}_Day'] = cleaned_df[selected_dt_col].dt.day
                        if extract_hour:
                            cleaned_df[f'{selected_dt_col}_Hour'] = cleaned_df[selected_dt_col].dt.hour
                        st.success(f"Features extracted from '{selected_dt_col}'.")


                st.session_state.cleaned_data = cleaned_df
                st.success("✅ Data cleaning and feature engineering completed!")
                
                # Show cleaning results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Shape", f"{df.shape[0]} × {df.shape[1]}")
                    st.metric("Original Missing Values", df.isnull().sum().sum())
                with col2:
                    st.metric("Cleaned Shape", f"{cleaned_df.shape[0]} × {cleaned_df.shape[1]}")
                    st.metric("Cleaned Missing Values", cleaned_df.isnull().sum().sum())
            
            # Preview cleaned data
            if st.session_state.cleaned_data is not None:
                st.markdown("### 👀 Cleaned Data Preview")
                st.dataframe(st.session_state.cleaned_data.head(10), use_container_width=True)
                
                # Download cleaned data
                st.markdown(get_download_link(st.session_state.cleaned_data, "cleaned_data.csv"), unsafe_allow_html=True)
        
        else:
            st.warning("⚠️ Please upload data first!")

    # DASHBOARD BUILDER PAGE
    elif page == "📈 Dashboard Builder":
        st.markdown('<h2 class="section-header">📈 Interactive Dashboard Builder</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
            
            # Get column types
            numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)
            
            # Dashboard configuration
            st.markdown("### ⚙️ Dashboard Configuration")
            
            # Chart selection
            chart_type = st.selectbox(
                "Select chart type:",
                ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Violin Plot", "Heatmap", "Pie Chart", "Choropleth Map"]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if chart_type in ["Scatter Plot", "Line Chart"]:
                    x_axis = st.selectbox("X-axis:", numeric_cols + categorical_cols + datetime_cols)
                    y_axis = st.selectbox("Y-axis:", numeric_cols)
                    color_by = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
                    
                elif chart_type in ["Bar Chart"]:
                    x_axis = st.selectbox("Category:", categorical_cols)
                    y_axis = st.selectbox("Value:", numeric_cols)
                    
                elif chart_type == "Histogram":
                    x_axis = st.selectbox("Column:", numeric_cols)
                    bins = st.slider("Number of bins:", 10, 100, 30)
                    
                elif chart_type in ["Box Plot", "Violin Plot"]:
                    y_axis = st.selectbox("Value column:", numeric_cols)
                    x_axis = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
                    
                elif chart_type == "Heatmap":
                    st.info("Correlation heatmap will be generated for numeric columns")
                    
                elif chart_type == "Pie Chart":
                    category_col = st.selectbox("Category column:", categorical_cols)

                elif chart_type == "Choropleth Map":
                    if categorical_cols and numeric_cols:
                        location_col = st.selectbox("Location Column (e.g., Country Name/Code):", categorical_cols)
                        value_col = st.selectbox("Value Column:", numeric_cols)
                        st.info("Ensure your location column contains country names or ISO alpha-3 codes.")
                    else:
                        st.warning("Need both categorical (for locations) and numeric (for values) columns for Choropleth Map.")
                        location_col = None
                        value_col = None
            
            with col2:
                # Chart customization
                st.markdown("#### 🎨 Customization")
                chart_title = st.text_input("Chart title:", f"{chart_type} Visualization")
                chart_height = st.slider("Chart height:", 300, 800, 500)
                color_scheme = st.selectbox("Color scheme:", ["plotly", "viridis", "plasma", "inferno", "magma", "cividis", "magma_r"])
                
                # For Choropleth, add scope
                if chart_type == "Choropleth Map":
                    map_scope = st.selectbox("Map Scope:", ["world", "asia", "europe", "africa", "north america", "south america"])


            # Generate chart
            if st.button("📊 Generate Chart"):
                try:
                    fig = None
                    
                    if chart_type == "Scatter Plot":
                        color = None if color_by == "None" else color_by
                        fig = px.scatter(df, x=x_axis, y=y_axis, color=color, 
                                         title=chart_title, height=chart_height, color_continuous_scale=color_scheme)
                    
                    elif chart_type == "Line Chart":
                        color = None if color_by == "None" else color_by
                        fig = px.line(df, x=x_axis, y=y_axis, color=color,
                                     title=chart_title, height=chart_height)
                    
                    elif chart_type == "Bar Chart":
                        df_grouped = df.groupby(x_axis)[y_axis].mean().reset_index()
                        fig = px.bar(df_grouped, x=x_axis, y=y_axis,
                                     title=chart_title, height=chart_height, color_discrete_sequence=px.colors.qualitative.Set3)
                    
                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=x_axis, nbins=bins,
                                             title=chart_title, height=chart_height, color_discrete_sequence=px.colors.qualitative.Set3)
                    
                    elif chart_type == "Box Plot":
                        x = None if x_axis == "None" else x_axis
                        fig = px.box(df, x=x, y=y_axis,
                                     title=chart_title, height=chart_height)

                    elif chart_type == "Violin Plot":
                        x = None if x_axis == "None" else x_axis
                        fig = px.violin(df, x=x, y=y_axis,
                                        title=chart_title, height=chart_height)
                    
                    elif chart_type == "Heatmap":
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                         title=chart_title, height=chart_height, color_continuous_scale=color_scheme)
                    
                    elif chart_type == "Pie Chart":
                        value_counts = df[category_col].value_counts()
                        fig = px.pie(values=value_counts.values, names=value_counts.index,
                                     title=chart_title, height=chart_height)

                    elif chart_type == "Choropleth Map":
                        if location_col and value_col:
                            fig = px.choropleth(df, locations=location_col, color=value_col, 
                                                hover_name=location_col, color_continuous_scale=color_scheme,
                                                title=chart_title, height=chart_height, scope=map_scope)
                        else:
                            st.warning("Please select valid location and value columns for Choropleth Map.")
                            fig = None
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional insights
                        st.markdown("### 💡 Quick Insights")
                        if chart_type == "Scatter Plot" and x_axis in numeric_cols and y_axis in numeric_cols:
                            correlation = df[x_axis].corr(df[y_axis])
                            st.write(f"📈 Correlation between {x_axis} and {y_axis}: {correlation:.3f}")
                        
                        elif chart_type == "Histogram":
                            st.write(f"📊 Mean: {df[x_axis].mean():.2f}, Median: {df[x_axis].median():.2f}, Std: {df[x_axis].std():.2f}")
                        
                except Exception as e:
                    st.error(f"❌ Error generating chart: {str(e)}")
            
            # Multi-chart dashboard
            st.markdown("---")
            st.markdown("### 📱 Multi-Chart Dashboard")
            
            if st.checkbox("Create multi-chart dashboard"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(numeric_cols) >= 2:
                        fig1 = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                         title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                        fig3 = px.box(df, x=categorical_cols[0], y=numeric_cols[0],
                                     title=f"{numeric_cols[0]} by {categorical_cols[0]}")
                        st.plotly_chart(fig3, use_container_width=True)
            
                with col2:
                    if len(numeric_cols) > 0:
                        fig2 = px.histogram(df, x=numeric_cols[0], 
                                             title=f"Distribution of {numeric_cols[0]}")
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    if len(numeric_cols) >= 2:
                        corr_matrix = df[numeric_cols].corr()
                        fig4 = px.imshow(corr_matrix, text_auto=True, 
                                         title="Correlation Heatmap")
                        st.plotly_chart(fig4, use_container_width=True)
        
        else:
            st.warning("⚠️ Please upload data first!")

    # MODEL BUILDER PAGE
    elif page == "🤖 Model Builder":
        st.markdown('<h2 class="section-header">🤖 Machine Learning Model Builder</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
            
            numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)
            all_cols = df.columns.tolist()
            
            st.markdown("### ⚙️ Model Configuration")
            
            # Problem Type
            problem_type = st.selectbox(
                "Select problem type:",
                ["Regression", "Classification", "Time Series Forecasting"] # Added Time Series Forecasting
            )
            
            # Conditional inputs based on problem type
            if problem_type in ["Regression", "Classification"]:
                # Target Variable
                target_variable = st.selectbox(
                    "Select target variable (Y):",
                    all_cols
                )
                
                # Ensure target variable is suitable for classification if selected
                if problem_type == "Classification":
                    if target_variable not in categorical_cols:
                        st.warning(f"Warning: '{target_variable}' is not a categorical column. Consider converting it or choosing a different target for classification.")
                        if st.checkbox("Convert target to categorical (Label Encoding)?"):
                            if df[target_variable].nunique() > 20: # Avoid encoding high cardinality columns
                                st.error("Too many unique values for label encoding. Please choose a different target or clean your data.")
                                st.stop()
                            le = LabelEncoder()
                            df[target_variable] = le.fit_transform(df[target_variable])
                            st.success(f"'{target_variable}' converted to numerical labels.")
                            # Update categorical columns list if this column was originally numeric
                            if target_variable in numeric_cols:
                                numeric_cols.remove(target_variable)
                            if target_variable not in categorical_cols:
                                categorical_cols.append(target_variable)
                    elif df[target_variable].nunique() < 2:
                        st.error("Target variable for classification must have at least two unique classes.")
                        st.stop()

                # Feature Variables
                available_features = [col for col in all_cols if col != target_variable]
                feature_variables = st.multiselect(
                    "Select feature variables (X):",
                    available_features,
                    default=available_features # Pre-select all by default
                )
                
                if not feature_variables:
                    st.warning("Please select at least one feature variable.")
                    st.stop()
                
                # Model Selection
                model_options = {
                    "Regression": ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"],
                    "Classification": ["Logistic Regression", "Random Forest Classifier", "XGBoost Classifier"]
                }
                
                selected_model = st.selectbox(
                    "Select a model:",
                    model_options[problem_type]
                )
                
                # Train-Test Split
                test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
                random_state = st.number_input("Random state:", 0, 100, 42)
                
                # Hyperparameter Tuning
                st.markdown("### 🧪 Hyperparameter Tuning (Grid Search)")
                perform_tuning = st.checkbox("Perform Hyperparameter Tuning?")
                
                if perform_tuning:
                    if selected_model in ["Random Forest Regressor", "Random Forest Classifier"]:
                        n_estimators_options = st.multiselect("n_estimators:", [50, 100, 200], default=[100])
                        max_depth_options = st.multiselect("max_depth:", [None, 10, 20], default=[None])
                        param_grid = {'n_estimators': n_estimators_options, 'max_depth': max_depth_options}
                    elif selected_model in ["XGBoost Regressor", "XGBoost Classifier"]:
                        n_estimators_options = st.multiselect("n_estimators:", [50, 100, 200], default=[100])
                        learning_rate_options = st.multiselect("learning_rate:", [0.01, 0.1, 0.2], default=[0.1])
                        param_grid = {'n_estimators': n_estimators_options, 'learning_rate': learning_rate_options}
                    else:
                        st.info("Hyperparameter tuning is currently supported for Random Forest and XGBoost models.")
                        param_grid = {}
                else:
                    param_grid = {} # Ensure param_grid is defined even if tuning is off

            else: # Time Series Forecasting
                st.markdown("### ⏰ Time Series Forecasting Configuration")
                if not datetime_cols:
                    st.warning("No datetime columns found in your data. Please ensure you have a datetime column for time series forecasting.")
                    st.stop()
                
                time_column = st.selectbox("Select Time Column:", datetime_cols)
                
                if not numeric_cols:
                    st.warning("No numeric columns found for time series target. Please ensure you have a numeric column.")
                    st.stop()
                target_variable_ts = st.selectbox("Select Target Variable (Y):", numeric_cols)
                
                forecast_horizon = st.number_input("Forecast Horizon (number of future steps):", min_value=1, value=10, step=1)
                
                st.markdown("#### ARIMA Model Orders (p, d, q)")
                p_order = st.number_input("ARIMA p (AR order):", min_value=0, value=5, step=1)
                d_order = st.number_input("ARIMA d (Differencing order):", min_value=0, value=1, step=1)
                q_order = st.number_input("ARIMA q (MA order):", min_value=0, value=0, step=1)

                selected_model = "ARIMA" # Fixed for time series for now
                feature_variables = [] # Not applicable in the same way for basic ARIMA
                param_grid = {} # No tuning for ARIMA in this basic setup
                test_size = 0.2 # Default for splitting if needed for evaluation of historical fit
                random_state = 42 # Default

            if st.button("🚀 Train Model"):
                try:
                    if problem_type in ["Regression", "Classification"]:
                        # Prepare data for Regression/Classification
                        X = df[feature_variables].copy()
                        y = df[target_variable].copy()
                        
                        # Store label encoders and scaler for prediction
                        label_encoders = {}
                        # Process categorical features in X
                        for col in X.select_dtypes(include='object').columns:
                            le = LabelEncoder()
                            # Fit and transform only non-null values to avoid errors, then fill back
                            non_null_values = X[col].dropna()
                            if not non_null_values.empty:
                                X.loc[non_null_values.index, col] = le.fit_transform(non_null_values)
                                label_encoders[col] = le
                            else:
                                st.warning(f"Column '{col}' has no non-null categorical values for Label Encoding.")
                        st.session_state.model_label_encoders = label_encoders
                        
                        # Convert all feature columns to numeric, coercing errors
                        for col in X.columns:
                            if not pd.api.types.is_numeric_dtype(X[col]):
                                original_dtype = X[col].dtype
                                X[col] = pd.to_numeric(X[col], errors='coerce')
                                if X[col].isnull().any():
                                    st.warning(f"Column '{col}' contained non-numeric values and was converted to numeric. Missing values introduced by coercion will be filled with the mean.")
                                    X[col] = X[col].fillna(X[col].mean() if not X[col].empty else 0) # Fill NaNs after coercion
                                if original_dtype != X[col].dtype:
                                    st.info(f"Column '{col}' transformed from {original_dtype} to {X[col].dtype}.")

                        # Handle any remaining NaNs in numeric columns (e.g., from original data or coercion)
                        for col in X.select_dtypes(include=[np.number]).columns:
                            if X[col].isnull().any():
                                st.warning(f"Missing values detected in numeric feature '{col}'. Filling with mean.")
                                X[col] = X[col].fillna(X[col].mean() if not X[col].empty else 0)

                        scaler = None
                        numeric_features_in_X = X.select_dtypes(include=[np.number]).columns
                        if not numeric_features_in_X.empty:
                            scaler = StandardScaler()
                            X[numeric_features_in_X] = scaler.fit_transform(X[numeric_features_in_X])
                        st.session_state.model_scaler = scaler
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                        
                        model = None
                        if selected_model == "Linear Regression":
                            model = LinearRegression()
                        elif selected_model == "Logistic Regression":
                            if y_train.dtype == 'object':
                                le_target = LabelEncoder()
                                y_train = le_target.fit_transform(y_train)
                                y_test = le_target.transform(y_test)
                                st.session_state.model_label_encoders[target_variable] = le_target # Store target encoder
                                st.info("Target variable for Logistic Regression was label encoded.")
                            model = LogisticRegression(max_iter=1000, random_state=random_state)
                        elif selected_model == "Random Forest Regressor":
                            model = RandomForestRegressor(random_state=random_state)
                        elif selected_model == "Random Forest Classifier":
                            if y_train.dtype == 'object':
                                le_target = LabelEncoder()
                                y_train = le_target.fit_transform(y_train)
                                y_test = le_target.transform(y_test)
                                st.session_state.model_label_encoders[target_variable] = le_target # Store target encoder
                                st.info("Target variable for Random Forest Classifier was label encoded.")
                            model = RandomForestClassifier(random_state=random_state)
                        elif selected_model == "XGBoost Regressor":
                            from xgboost import XGBRegressor
                            model = XGBRegressor(random_state=random_state)
                        elif selected_model == "XGBoost Classifier":
                            from xgboost import XGBClassifier
                            if y_train.dtype == 'object':
                                le_target = LabelEncoder()
                                y_train = le_target.fit_transform(y_train)
                                y_test = le_target.transform(y_test)
                                st.session_state.model_label_encoders[target_variable] = le_target # Store target encoder
                                st.info("Target variable for XGBoost Classifier was label encoded.")
                            model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
                        
                        if model:
                            if perform_tuning and param_grid:
                                st.info(f"Performing Grid Search for {selected_model}...")
                                scoring_metric = 'neg_mean_squared_error' if problem_type == 'Regression' else 'accuracy'
                                grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring_metric, n_jobs=-1)
                                grid_search.fit(X_train, y_train)
                                model = grid_search.best_estimator_
                                st.success(f"Best parameters found: {grid_search.best_params_}")
                                st.info(f"Training {selected_model} with best parameters...")
                            else:
                                st.info(f"Training {selected_model}...")
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            st.success("✅ Model training complete!")
                            
                            st.markdown("### 📈 Model Evaluation")
                            evaluation_metrics = {}
                            if problem_type == "Regression":
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y_test, y_pred)
                                st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
                                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
                                st.metric("R-squared (R2)", f"{r2:.2f}")
                                evaluation_metrics = {'mse': mse, 'rmse': rmse, 'r2': r2}
                                
                                # Plot actual vs. predicted (for regression)
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual vs. Predicted'))
                                fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', name='Ideal Line', 
                                                         line=dict(color='red', dash='dash')))
                                fig.update_layout(title="Actual vs. Predicted Values", xaxis_title="Actual Values", yaxis_title="Predicted Values")
                                st.plotly_chart(fig, use_container_width=True)

                            elif problem_type == "Classification":
                                accuracy = accuracy_score(y_test, y_pred)
                                st.metric("Accuracy Score", f"{accuracy:.2f}")
                                st.subheader("Classification Report")
                                report = classification_report(y_test, y_pred, output_dict=True)
                                st.text(classification_report(y_test, y_pred))
                                evaluation_metrics = {'accuracy': accuracy, 'classification_report': report}

                                # Confusion Matrix (using Plotly for interactivity)
                                st.subheader("Confusion Matrix")
                                cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
                                
                                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                                    labels=dict(x="Predicted", y="Actual", color="Count"),
                                                    x=cm.columns.tolist(), y=cm.index.tolist())
                                fig_cm.update_layout(title_text='Confusion Matrix')
                                st.plotly_chart(fig_cm, use_container_width=True)
                            
                            st.session_state.models[selected_model] = {
                                'model': model,
                                'features': feature_variables,
                                'target': target_variable,
                                'problem_type': problem_type,
                                'evaluation': evaluation_metrics
                            }
                            # Store the latest trained model for the prediction interface
                            st.session_state.trained_model = model
                            st.session_state.model_features = feature_variables
                            st.session_state.model_target = target_variable
                            st.session_state.model_problem_type = problem_type
                            st.session_state.model_evaluation_metrics = evaluation_metrics # Store metrics for display
                            st.info(f"Model '{selected_model}' saved in session state.")
                    
                    elif problem_type == "Time Series Forecasting":
                        # Prepare data for Time Series
                        ts_data = df[[time_column, target_variable_ts]].copy()
                        ts_data[time_column] = pd.to_datetime(ts_data[time_column], errors='coerce')
                        ts_data.set_index(time_column, inplace=True)
                        ts_data.dropna(subset=[target_variable_ts], inplace=True) # Drop rows where target is NaN

                        if ts_data.empty:
                            st.error("Time series data is empty after processing. Cannot train model.")
                            st.stop()
                        
                        # Split data into train and test for evaluation
                        train_size = int(len(ts_data) * (1 - test_size))
                        train_data, test_data = ts_data.iloc[:train_size], ts_data.iloc[train_size:]

                        st.info(f"Training ARIMA({p_order}, {d_order}, {q_order}) model...")
                        model = ARIMA(train_data[target_variable_ts], order=(p_order, d_order, q_order))
                        model_fit = model.fit()
                        
                        st.success("✅ ARIMA model training complete!")
                        st.subheader("Model Summary")
                        st.text(model_fit.summary())

                        # Make in-sample predictions for evaluation
                        y_pred_train = model_fit.predict(start=0, end=len(train_data)-1)
                        rmse_train = np.sqrt(mean_squared_error(train_data[target_variable_ts], y_pred_train))
                        mae_train = mean_absolute_error(train_data[target_variable_ts], y_pred_train)
                        st.write(f"**In-sample RMSE:** {rmse_train:.2f}")
                        st.write(f"**In-sample MAE:** {mae_train:.2f}")

                        # Forecast future values
                        # Use get_forecast for more robust future prediction with confidence intervals
                        forecast_result = model_fit.get_forecast(steps=forecast_horizon)
                        forecast_values = forecast_result.predicted_mean
                        conf_int = forecast_result.conf_int()

                        # Create future index for plotting
                        last_date = ts_data.index[-1]
                        future_index = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=pd.infer_freq(ts_data.index))[1:] # Exclude last date, add forecast horizon

                        forecast_series = pd.Series(forecast_values.values, index=future_index)
                        lower_bound = pd.Series(conf_int.iloc[:, 0].values, index=future_index)
                        upper_bound = pd.Series(conf_int.iloc[:, 1].values, index=future_index)

                        st.markdown("### 📈 Time Series Forecast")
                        
                        fig_ts = go.Figure()
                        fig_ts.add_trace(go.Scatter(x=ts_data.index, y=ts_data[target_variable_ts], mode='lines', name='Historical Data'))
                        fig_ts.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines', name='Forecast', line=dict(color='red')))
                        fig_ts.add_trace(go.Scatter(x=lower_bound.index, y=lower_bound, mode='lines', name='Lower Bound (95% CI)', line=dict(color='grey', dash='dash'), showlegend=False))
                        fig_ts.add_trace(go.Scatter(x=upper_bound.index, y=upper_bound, mode='lines', name='Upper Bound (95% CI)', fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='grey', dash='dash')))
                        
                        fig_ts.update_layout(title=f"Time Series Forecast for {target_variable_ts}",
                                             xaxis_title="Date",
                                             yaxis_title=target_variable_ts)
                        st.plotly_chart(fig_ts, use_container_width=True)

                        st.subheader("Forecasted Values")
                        forecast_df = pd.DataFrame({
                            'Date': forecast_series.index,
                            'Forecast': forecast_series.values,
                            'Lower CI': lower_bound.values,
                            'Upper CI': upper_bound.values
                        })
                        st.dataframe(forecast_df, use_container_width=True)

                        # Store model and relevant info for saving/loading
                        st.session_state.trained_model = model_fit # Store the fitted model
                        st.session_state.model_features = [time_column] # Time column is the 'feature'
                        st.session_state.model_target = target_variable_ts
                        st.session_state.model_problem_type = problem_type
                        st.session_state.model_scaler = None # No scaler for ARIMA directly on target
                        st.session_state.model_label_encoders = {} # No label encoders for ARIMA
                        st.session_state.model_evaluation_metrics = {
                            'in_sample_rmse': rmse_train,
                            'in_sample_mae': mae_train,
                            'forecast_horizon': forecast_horizon,
                            'arima_order': (p_order, d_order, q_order)
                        }
                        st.info(f"Model '{selected_model}' saved in session state.")

                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
        else:
            st.warning("⚠️ Please upload data first!")

        # Model Saving
        st.markdown("---")
        st.markdown("### 💾 Save Trained Model")
        if st.session_state.trained_model is not None:
            model_filename = st.text_input("Enter a filename for your model (e.g., my_regression_model.joblib):", 
                                            value=f"{st.session_state.model_problem_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.joblib")
            if st.button("Save Model"):
                try:
                    model_data = {
                        'model': st.session_state.trained_model,
                        'features': st.session_state.model_features,
                        'target': st.session_state.model_target,
                        'problem_type': st.session_state.model_problem_type,
                        'scaler': st.session_state.model_scaler,
                        'label_encoders': st.session_state.model_label_encoders,
                        'evaluation_metrics': st.session_state.model_evaluation_metrics
                    }
                    save_path = os.path.join('saved_models', model_filename)
                    joblib.dump(model_data, save_path)
                    st.success(f"Model saved successfully to `{save_path}`")
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
        else:
            st.info("Train a model first to enable saving.")

        # Model Loading
        st.markdown("---")
        st.markdown("### 📂 Load Saved Model")
        saved_model_files = [f for f in os.listdir('saved_models') if f.endswith('.joblib')]
        if saved_model_files:
            selected_load_model = st.selectbox("Select a model to load:", ["None"] + saved_model_files)
            if selected_load_model != "None" and st.button("Load Model"):
                try:
                    load_path = os.path.join('saved_models', selected_load_model)
                    model_data = joblib.load(load_path)
                    
                    st.session_state.trained_model = model_data['model']
                    st.session_state.model_features = model_data['features']
                    st.session_state.model_target = model_data['target']
                    st.session_state.model_problem_type = model_data['problem_type']
                    st.session_state.model_scaler = model_data['scaler']
                    st.session_state.model_label_encoders = model_data['label_encoders']
                    st.session_state.model_evaluation_metrics = model_data.get('evaluation_metrics', {}) # Get with default for backward compatibility

                    st.success(f"Model '{selected_load_model}' loaded successfully!")
                    st.info(f"Loaded model type: **{type(st.session_state.trained_model).__name__}**")
                    st.info(f"Target variable: **{st.session_state.model_target}**")
                    st.info(f"Features used: **{', '.join(st.session_state.model_features)}**")

                    st.markdown("#### Loaded Model Evaluation Metrics (from training time):")
                    metrics = st.session_state.model_evaluation_metrics
                    if metrics:
                        if st.session_state.model_problem_type == "Regression":
                            if 'mse' in metrics: st.metric("MSE", f"{metrics['mse']:.2f}")
                            if 'rmse' in metrics: st.metric("RMSE", f"{metrics['rmse']:.2f}")
                            if 'r2' in metrics: st.metric("R-squared", f"{metrics['r2']:.2f}")
                        elif st.session_state.model_problem_type == "Classification":
                            if 'accuracy' in metrics: st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
                            if 'classification_report' in metrics:
                                st.subheader("Classification Report")
                                # Convert dict report back to string for display if needed, or pretty print
                                if isinstance(metrics['classification_report'], dict):
                                    report_str = ""
                                    for k, v in metrics['classification_report'].items():
                                        if isinstance(v, dict):
                                            report_str += f"{k}:\n"
                                            for sub_k, sub_v in v.items():
                                                if isinstance(sub_v, float):
                                                    report_str += f"  {sub_k}: {sub_v:.2f}\n"
                                                else:
                                                    report_str += f"  {sub_k}: {sub_v}\n"
                                        else:
                                            report_str += f"{k}: {v}\n"
                                    st.text(report_str)
                                else:
                                    st.text(metrics['classification_report'])
                        elif st.session_state.model_problem_type == "Time Series Forecasting":
                            st.metric("In-sample RMSE", f"{metrics.get('in_sample_rmse', 0):.2f}")
                            st.metric("In-sample MAE", f"{metrics.get('in_sample_mae', 0):.2f}")
                            st.write(f"Forecast Horizon: {metrics.get('forecast_horizon', 'N/A')} steps")
                            st.write(f"ARIMA Order: {metrics.get('arima_order', 'N/A')}")
                    else:
                        st.info("No evaluation metrics saved with this model.")

                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        else:
            st.info("No saved models found. Train and save a model first!")


        # Prediction Interface (remains the same, but now uses loaded model)
        st.markdown("---")
        st.markdown("### 🔮 Make Predictions")
        if st.session_state.trained_model is not None:
            st.info(f"Using trained model: **{type(st.session_state.trained_model).__name__}** to predict **{st.session_state.model_target}**.")
            
            if st.session_state.model_problem_type in ["Regression", "Classification"]:
                input_data = {}
                col_idx = 0
                cols_per_row = 3 # Adjust as needed for layout

                # Create input fields for each feature
                for feature in st.session_state.model_features:
                    if col_idx % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                    
                    with cols[col_idx % cols_per_row]:
                        # Determine input type based on original column type (before encoding/scaling)
                        # Use the original df for type inference, not the cleaned_df, as cleaned_df might have OHE etc.
                        original_dtype = st.session_state.data[feature].dtype if st.session_state.data is not None else None
                        
                        if pd.api.types.is_numeric_dtype(original_dtype):
                            input_data[feature] = st.number_input(f"Enter {feature}:", value=float(st.session_state.data[feature].mean()) if st.session_state.data is not None and not st.session_state.data[feature].empty else 0.0, key=f"pred_input_{feature}")
                        elif pd.api.types.is_object_dtype(original_dtype) or pd.api.types.is_categorical_dtype(original_dtype):
                            # Use the stored label encoder to get original classes
                            if feature in st.session_state.model_label_encoders:
                                le = st.session_state.model_label_encoders[feature]
                                options = list(le.classes_)
                                input_data[feature] = st.selectbox(f"Select {feature}:", options, key=f"pred_input_{feature}")
                            else:
                                # Fallback if encoder not found (shouldn't happen if handled correctly)
                                input_data[feature] = st.text_input(f"Enter {feature} (Categorical):", key=f"pred_input_{feature}")
                        else: # Default to text input for other types
                            input_data[feature] = st.text_input(f"Enter {feature}:", key=f"pred_input_{feature}")
                    col_idx += 1

                if st.button("Get Prediction"):
                    try:
                        # Create a DataFrame from input data
                        input_df = pd.DataFrame([input_data])

                        # Apply transformations (encoding, scaling) using stored objects
                        for col in st.session_state.model_features: # Iterate through features used by the model
                            if col in st.session_state.model_label_encoders:
                                le = st.session_state.model_label_encoders[col]
                                # Check if the column is in the input_df and if it's an object/categorical type
                                if col in input_df.columns and (pd.api.types.is_object_dtype(input_df[col].dtype) or pd.api.types.is_categorical_dtype(input_df[col].dtype)):
                                    # Handle unseen labels: if the input value is not in known classes, use a default (e.g., mode of original data)
                                    if input_df[col].iloc[0] not in le.classes_:
                                        st.warning(f"Input value '{input_df[col].iloc[0]}' for '{col}' not seen during training. Using mode for encoding.")
                                        # Find the mode from the original training data for this column
                                        original_col_mode = st.session_state.data[col].mode().iloc[0] if st.session_state.data is not None and not st.session_state.data[col].empty else le.classes_[0] # Fallback to first class
                                        input_df[col] = le.transform([original_col_mode])
                                    else:
                                        input_df[col] = le.transform(input_df[col])
                                elif col in input_df.columns: # If it's a numeric column that was encoded (e.g., if it was originally categorical but became numeric after OHE)
                                    # This case is tricky if OHE was applied. For now, assume LabelEncoder was only for categorical.
                                    pass 
                        
                        if st.session_state.model_scaler:
                            numeric_features_to_scale = input_df.select_dtypes(include=[np.number]).columns
                            if not numeric_features_to_scale.empty:
                                input_df[numeric_features_to_scale] = st.session_state.model_scaler.transform(input_df[numeric_features_to_scale])
                        
                        prediction = st.session_state.trained_model.predict(input_df[st.session_state.model_features])
                        
                        if st.session_state.model_problem_type == "Classification" and st.session_state.model_target in st.session_state.model_label_encoders:
                            # Inverse transform the prediction if the target was encoded
                            le_target = st.session_state.model_label_encoders[st.session_state.model_target]
                            predicted_label = le_target.inverse_transform(prediction)
                            st.success(f"Predicted {st.session_state.model_target}: **{predicted_label[0]}**")
                        else:
                            st.success(f"Predicted {st.session_state.model_target}: **{prediction[0]:.2f}**")

                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
            elif st.session_state.model_problem_type == "Time Series Forecasting":
                st.info("For time series models, predictions are made for a future horizon defined during training/loading.")
                st.write(f"The loaded ARIMA model was trained to forecast **{st.session_state.model_evaluation_metrics.get('forecast_horizon', 'N/A')}** steps into the future.")
                
                if st.button("Generate Time Series Forecast"):
                    try:
                        # Ensure the loaded model is an ARIMA model
                        if not isinstance(st.session_state.trained_model, ARIMA):
                            st.error("The loaded model is not an ARIMA model. Please load a valid ARIMA model for time series forecasting.")
                            st.stop()

                        # Re-create the time series data for the loaded model's target and time column
                        # This assumes the original data is still in session_state.data
                        if st.session_state.data is None:
                            st.error("Original data not found in session. Please upload data first to generate time series forecasts.")
                            st.stop()

                        time_col = st.session_state.model_features[0] # Assuming time column is the first feature
                        target_col = st.session_state.model_target

                        ts_data_full = st.session_state.data[[time_col, target_col]].copy()
                        ts_data_full[time_col] = pd.to_datetime(ts_data_full[time_col], errors='coerce')
                        ts_data_full.set_index(time_col, inplace=True)
                        ts_data_full.dropna(subset=[target_col], inplace=True)

                        if ts_data_full.empty:
                            st.error("Time series data is empty after processing. Cannot generate forecast.")
                            st.stop()
                        
                        # Use the loaded model to forecast
                        forecast_horizon = st.session_state.model_evaluation_metrics.get('forecast_horizon', 10)
                        
                        # Use the loaded model's `predict` or `forecast` method
                        # For ARIMA, `get_forecast` is preferred for confidence intervals
                        forecast_result = st.session_state.trained_model.get_forecast(steps=forecast_horizon)
                        forecast_values = forecast_result.predicted_mean
                        conf_int = forecast_result.conf_int()

                        # Create future index for plotting
                        last_date = ts_data_full.index[-1]
                        future_index = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=pd.infer_freq(ts_data_full.index))[1:]

                        forecast_series = pd.Series(forecast_values.values, index=future_index)
                        lower_bound = pd.Series(conf_int.iloc[:, 0].values, index=future_index)
                        upper_bound = pd.Series(conf_int.iloc[:, 1].values, index=future_index)

                        st.markdown("### 📈 Time Series Forecast")
                        
                        fig_ts = go.Figure()
                        fig_ts.add_trace(go.Scatter(x=ts_data_full.index, y=ts_data_full[target_col], mode='lines', name='Historical Data'))
                        fig_ts.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines', name='Forecast', line=dict(color='red')))
                        fig_ts.add_trace(go.Scatter(x=lower_bound.index, y=lower_bound, mode='lines', name='Lower Bound (95% CI)', line=dict(color='grey', dash='dash'), showlegend=False))
                        fig_ts.add_trace(go.Scatter(x=upper_bound.index, y=upper_bound, mode='lines', name='Upper Bound (95% CI)', fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='grey', dash='dash')))
                        
                        fig_ts.update_layout(title=f"Time Series Forecast for {target_col}",
                                             xaxis_title="Date",
                                             yaxis_title=target_col)
                        st.plotly_chart(fig_ts, use_container_width=True)

                        st.subheader("Forecasted Values")
                        forecast_df = pd.DataFrame({
                            'Date': forecast_series.index,
                            'Forecast': forecast_series.values,
                            'Lower CI': lower_bound.values,
                            'Upper CI': upper_bound.values
                        })
                        st.dataframe(forecast_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error generating time series forecast: {str(e)}")
        else:
            st.info("Train or load a model first to enable the prediction interface.")


    # ADVANCED ANALYTICS PAGE
    elif page == "📊 Advanced Analytics":
        st.markdown('<h2 class="section-header">📊 Advanced Statistical Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
            numeric_cols, categorical_cols, _ = detect_column_types(df)
            
            st.markdown("### 🔍 Data Distribution & Statistics")
            
            if not numeric_cols:
                st.info("No numeric columns found for distribution analysis. Please ensure your data has numeric columns or clean them appropriately.")
            else:
                selected_numeric_col = st.selectbox("Select a numeric column for analysis:", numeric_cols)
                
                if selected_numeric_col:
                    st.subheader(f"Statistics for '{selected_numeric_col}'")
                    st.dataframe(df[selected_numeric_col].describe().to_frame(), use_container_width=True)
                    
                    fig = px.histogram(df, x=selected_numeric_col, marginal="box", 
                                       title=f"Distribution of {selected_numeric_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📉 Correlation Analysis")
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                     title="Correlation Matrix of Numeric Features",
                                     color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Need at least two numeric columns for correlation analysis.")
            
            st.markdown("---")
            st.markdown("### 📊 Grouped Statistics")
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                group_by_col = st.selectbox("Group by (categorical column):", categorical_cols)
                agg_col = st.selectbox("Aggregate (numeric column):", numeric_cols)
                agg_method = st.selectbox("Aggregation method:", ["mean", "median", "sum", "count", "std"])
                
                if st.button("Generate Grouped Statistics"):
                    grouped_data = df.groupby(group_by_col)[agg_col].agg(agg_method).reset_index()
                    st.dataframe(grouped_data, use_container_width=True)
                    
                    fig_grouped = px.bar(grouped_data, x=group_by_col, y=agg_col,
                                         title=f"{agg_method.capitalize()} of {agg_col} by {group_by_col}")
                    st.plotly_chart(fig_grouped, use_container_width=True)
            else:
                st.info("Need at least one categorical and one numeric column for grouped statistics.")

            st.markdown("---")
            st.markdown("### 🔬 Hypothesis Testing (Two-sample T-test)")
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                t_test_numeric_col = st.selectbox("Select Numeric Column:", numeric_cols, key="t_test_num")
                t_test_group_col = st.selectbox("Select Grouping (Categorical) Column:", categorical_cols, key="t_test_cat")

                if t_test_numeric_col and t_test_group_col:
                    unique_groups = df[t_test_group_col].dropna().unique()
                    if len(unique_groups) >= 2:
                        group1 = st.selectbox("Select Group 1:", unique_groups, key="group1")
                        group2 = st.selectbox("Select Group 2:", [g for g in unique_groups if g != group1], key="group2")
                        
                        if st.button("Perform T-test"):
                            data_group1 = df[df[t_test_group_col] == group1][t_test_numeric_col].dropna()
                            data_group2 = df[df[t_test_group_col] == group2][t_test_numeric_col].dropna()

                            if len(data_group1) > 1 and len(data_group2) > 1:
                                t_stat, p_value = stats.ttest_ind(data_group1, data_group2, equal_var=False) # Welch's t-test
                                st.write(f"**Comparing means of '{t_test_numeric_col}' between '{group1}' and '{group2}'**")
                                st.write(f"T-statistic: {t_stat:.3f}")
                                st.write(f"P-value: {p_value:.3f}")

                                alpha = 0.05
                                if p_value < alpha:
                                    st.success(f"Result: Reject the null hypothesis. There is a statistically significant difference between the means (p < {alpha}).")
                                else:
                                    st.info(f"Result: Fail to reject the null hypothesis. No statistically significant difference found (p >= {alpha}).")
                            else:
                                st.warning("Not enough data points in one or both groups to perform T-test.")
                    else:
                        st.info("Grouping column must have at least two unique values for T-test.")
            else:
                st.info("Need both numeric and categorical columns for T-test.")

            st.markdown("---")
            st.markdown("### 📊 Principal Component Analysis (PCA)")
            if len(numeric_cols) >= 2:
                pca_cols = st.multiselect("Select numeric columns for PCA:", numeric_cols, default=numeric_cols)
                
                if pca_cols:
                    n_components = st.slider("Number of components:", 1, min(len(pca_cols), 10), min(len(pca_cols), 2))
                    
                    if st.button("Perform PCA"):
                        try:
                            pca_df = df[pca_cols].dropna()
                            if pca_df.empty:
                                st.warning("Selected columns contain too many missing values for PCA. Please clean data first.")
                            else:
                                scaler_pca = StandardScaler()
                                scaled_data = scaler_pca.fit_transform(pca_df)
                                
                                pca = PCA(n_components=n_components)
                                principal_components = pca.fit_transform(scaled_data)
                                
                                explained_variance_ratio = pca.explained_variance_ratio_
                                cumulative_explained_variance = np.cumsum(explained_variance_ratio)

                                st.subheader("Explained Variance Ratio")
                                fig_exp_var = px.bar(
                                    x=[f'PC{i+1}' for i in range(len(explained_variance_ratio))],
                                    y=explained_variance_ratio,
                                    labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
                                    title='Explained Variance Ratio by Principal Component'
                                )
                                st.plotly_chart(fig_exp_var, use_container_width=True)

                                st.subheader("Cumulative Explained Variance")
                                fig_cum_exp_var = px.line(
                                    x=[f'PC{i+1}' for i in range(len(cumulative_explained_variance))],
                                    y=cumulative_explained_variance,
                                    labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'},
                                    title='Cumulative Explained Variance'
                                )
                                fig_cum_exp_var.add_trace(go.Scatter(x=[f'PC{i+1}' for i in range(len(cumulative_explained_variance))], y=[0.95]*len(cumulative_explained_variance), mode='lines', name='95% Threshold', line=dict(dash='dash', color='red')))
                                st.plotly_chart(fig_cum_exp_var, use_container_width=True)

                                st.write(f"Total explained variance with {n_components} components: {cumulative_explained_variance[-1]:.2f}")
                                
                                # Display PCA components if n_components <= 3
                                if n_components <= 3:
                                    pca_df_transformed = pd.DataFrame(data=principal_components, 
                                                                    columns=[f'PC{i+1}' for i in range(n_components)])
                                    st.subheader("Transformed Data (First few Principal Components)")
                                    st.dataframe(pca_df_transformed.head(), use_container_width=True)
                                    if n_components >= 2:
                                        fig_pca = px.scatter(pca_df_transformed, x='PC1', y='PC2', 
                                                            title='PCA: PC1 vs PC2')
                                        st.plotly_chart(fig_pca, use_container_width=True)
                                    if n_components == 3:
                                        fig_pca_3d = px.scatter_3d(pca_df_transformed, x='PC1', y='PC2', z='PC3',
                                                                    title='PCA: PC1 vs PC2 vs PC3')
                                        st.plotly_chart(fig_pca_3d, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error performing PCA: {str(e)}")
                else:
                    st.info("Please select columns for PCA.")
            else:
                st.info("Need at least two numeric columns to perform PCA.")

            st.markdown("---")
            st.markdown("### ➕ K-Means Clustering")
            if len(numeric_cols) >= 2:
                kmeans_cols = st.multiselect("Select numeric columns for K-Means Clustering:", numeric_cols)
                if kmeans_cols:
                    n_clusters = st.slider("Number of clusters (K):", 2, 10, 3)
                    
                    if st.button("Run K-Means Clustering"):
                        try:
                            kmeans_df = df[kmeans_cols].dropna()
                            if kmeans_df.empty:
                                st.warning("Selected columns contain too many missing values for K-Means. Please clean data first.")
                            else:
                                scaler_kmeans = StandardScaler()
                                scaled_data_kmeans = scaler_kmeans.fit_transform(kmeans_df)
                                
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Added n_init
                                cluster_labels = kmeans.fit_predict(scaled_data_kmeans)
                                
                                kmeans_df['Cluster'] = cluster_labels
                                
                                st.success(f"✅ K-Means clustering completed with {n_clusters} clusters!")
                                st.write("### Cluster Assignments Preview")
                                st.dataframe(kmeans_df.head(), use_container_width=True)

                                st.write("### Cluster Counts")
                                st.dataframe(kmeans_df['Cluster'].value_counts().sort_index())
                                
                                st.write("### Cluster Centroids (Scaled Features)")
                                centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=kmeans_cols)
                                st.dataframe(centroids_df)

                                # Visualization of clusters (using first two selected features if available)
                                if len(kmeans_cols) >= 2:
                                    fig_kmeans = px.scatter(kmeans_df, x=kmeans_cols[0], y=kmeans_cols[1], color='Cluster',
                                                            title=f"K-Means Clusters ({kmeans_cols[0]} vs {kmeans_cols[1]})",
                                                            color_continuous_scale=px.colors.qualitative.Plotly)
                                    # Add centroids to the plot
                                    centroids_df['Cluster'] = centroids_df.index
                                    fig_kmeans.add_trace(go.Scatter(x=centroids_df[kmeans_cols[0]], y=centroids_df[kmeans_cols[1]],
                                                                    mode='markers',
                                                                    marker=dict(symbol='x', size=15, color='red', line=dict(width=2, color='DarkSlateGrey')),
                                                                    name='Centroids'))
                                    st.plotly_chart(fig_kmeans, use_container_width=True)
                                else:
                                    st.info("Select at least two columns to visualize K-Means clusters on a scatter plot.")

                                st.markdown("### Elbow Method for Optimal K")
                                if len(kmeans_cols) > 0:
                                    max_k = min(10, len(kmeans_df) // 2) # Prevent K from being too large for small datasets
                                    if max_k < 2:
                                        st.warning("Not enough data points to plot Elbow Method for K-Means.")
                                    else:
                                        inertia_values = []
                                        k_range = range(1, max_k + 1)
                                        progress_bar = st.progress(0)
                                        for i, k in enumerate(k_range):
                                            progress_bar.progress((i + 1) / len(k_range))
                                            kmeans_elbow = KMeans(n_clusters=k, random_state=42, n_init=10)
                                            kmeans_elbow.fit(scaled_data_kmeans)
                                            inertia_values.append(kmeans_elbow.inertia_)
                                        progress_bar.empty()
                                        
                                        fig_elbow = px.line(x=list(k_range), y=inertia_values, markers=True,
                                                            title='Elbow Method for Optimal K',
                                                            labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'})
                                        st.plotly_chart(fig_elbow, use_container_width=True)
                                        st.info("Look for the 'elbow' point in the graph where the rate of decrease in inertia slows down significantly.")

                        except Exception as e:
                            st.error(f"❌ Error performing K-Means Clustering: {str(e)}")
                else:
                    st.info("Please select columns for K-Means Clustering.")
            else:
                st.info("Need at least two numeric columns for K-Means Clustering.")
            
            st.markdown("---")
            st.markdown("### 📉 Dimensionality Reduction & Visualization (t-SNE)")
            if len(numeric_cols) >= 2:
                tsne_cols = st.multiselect("Select numeric columns for t-SNE (computationally intensive for large datasets):", numeric_cols)
                if tsne_cols:
                    tsne_n_components = st.radio("Number of dimensions for t-SNE output:", [2, 3])
                    
                    if st.button("Run t-SNE Visualization"):
                        try:
                            tsne_df = df[tsne_cols].dropna()
                            if tsne_df.empty:
                                st.warning("Selected columns contain too many missing values for t-SNE. Please clean data first.")
                            else:
                                if len(tsne_df) > 5000: # Warn for very large datasets
                                    st.warning(f"Warning: t-SNE can be slow for datasets larger than 5000 rows. Your dataset has {len(tsne_df)} rows. Consider sampling if it takes too long.")

                                scaler_tsne = StandardScaler()
                                scaled_data_tsne = scaler_tsne.fit_transform(tsne_df)
                                
                                tsne = TSNE(n_components=tsne_n_components, random_state=42, n_jobs=-1, perplexity=min(30, len(scaled_data_tsne) - 1)) # Added n_jobs, adjusted perplexity
                                tsne_transformed_data = tsne.fit_transform(scaled_data_tsne)
                                
                                tsne_results_df = pd.DataFrame(tsne_transformed_data, columns=[f'TSNE_Component_{i+1}' for i in range(tsne_n_components)])
                                
                                st.success(f"✅ t-SNE dimensionality reduction completed to {tsne_n_components}D!")
                                st.dataframe(tsne_results_df.head(), use_container_width=True)

                                if tsne_n_components == 2:
                                    fig_tsne = px.scatter(tsne_results_df, x='TSNE_Component_1', y='TSNE_Component_2',
                                                          title='t-SNE 2D Visualization',
                                                          hover_data=tsne_cols # Show original values on hover
                                                         )
                                    st.plotly_chart(fig_tsne, use_container_width=True)
                                elif tsne_n_components == 3:
                                    fig_tsne_3d = px.scatter_3d(tsne_results_df, x='TSNE_Component_1', y='TSNE_Component_2', z='TSNE_Component_3',
                                                                title='t-SNE 3D Visualization',
                                                                hover_data=tsne_cols # Show original values on hover
                                                               )
                                    st.plotly_chart(fig_tsne_3d, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"❌ Error performing t-SNE: {str(e)}")
                else:
                    st.info("Please select columns for t-SNE.")
            else:
                st.info("Need at least two numeric columns for t-SNE.")
                
        else:
            st.warning("⚠️ Please upload data first!")

    # BUSINESS INTELLIGENCE PAGE
    elif page == "📈 Business Intelligence":
        st.markdown('<h2 class="section-header">📈 Business Intelligence Features</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
            numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)

            st.markdown("### 📊 Advanced Reporting")
            st.write("This section provides tools for creating and managing advanced reports.")
            
            st.subheader("Automated Report Generation")
            st.info("Configure automated generation of summary reports based on your data.")
            if st.button("Generate Sample Summary Report"):
                if not df.empty:
                    st.write("#### Sample Report Data:")
                    st.dataframe(df.describe(), use_container_width=True)
                    st.success("Sample summary report generated. In a full BI system, this would be a downloadable PDF/Excel.")
                else:
                    st.warning("No data available to generate a report. Please upload data first.")

            st.subheader("Executive Dashboards and KPI Monitoring")
            st.info("Monitor key performance indicators (KPIs) and visualize executive-level dashboards.")
            if st.button("View Sample KPI Dashboard"):
                if not df.empty and numeric_cols:
                    st.write("#### Sample KPI Dashboard (using first numeric column as a KPI):")
                    kpi_col = numeric_cols[0]
                    avg_kpi = df[kpi_col].mean()
                    max_kpi = df[kpi_col].max()
                    min_kpi = df[kpi_col].min()

                    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
                    with col_kpi1:
                        st.metric(f"Average {kpi_col}", f"{avg_kpi:.2f}")
                    with col_kpi2:
                        st.metric(f"Max {kpi_col}", f"{max_kpi:.2f}")
                    with col_kpi3:
                        st.metric(f"Min {kpi_col}", f"{min_kpi:.2f}")
                    
                    fig_kpi = px.line(df, y=kpi_col, title=f"Trend of {kpi_col}")
                    st.plotly_chart(fig_kpi, use_container_width=True)
                else:
                    st.warning("No numeric data available for KPI monitoring. Please upload data with numeric columns.")

            st.subheader("Scheduled Report Delivery")
            st.info("Set up schedules for automatic delivery of reports via email or other channels.")
            st.markdown("- *This feature would typically integrate with external scheduling services (e.g., cron jobs, cloud functions) and email APIs.*")

            st.subheader("Custom Report Templates")
            st.info("Design and manage custom templates for your reports to ensure consistent branding and layout.")
            st.markdown("- *This would involve a template editor or upload functionality, allowing users to define report structures.*")

            st.subheader("Interactive Storytelling with Data")
            st.info("Create guided, interactive data narratives to present insights effectively.")
            st.markdown("- *This feature could leverage Streamlit's multi-page capabilities or custom components to build sequential data stories.*")

            st.markdown("---")
            st.info("To fully implement these BI features, integration with backend services for scheduling, email, and more complex dashboarding frameworks would be required.")
        else:
            st.warning("⚠️ Please upload data first to explore Business Intelligence features!")

    # FINANCIAL ANALYTICS PAGE
    elif page == "💰 Financial Analytics":
        st.markdown('<h2 class="section-header">💰 Financial Analytics</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
            numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)

            st.markdown("### 📈 Risk Modeling and Portfolio Optimization")
            st.info("Develop and analyze risk models, and optimize investment portfolios.")
            st.markdown("- *This would involve advanced financial modeling libraries (e.g., `PyPortfolioOpt`, `QuantLib`) and optimization algorithms.*")
            if st.button("Simulate Basic Portfolio Returns"):
                if len(numeric_cols) >= 2:
                    st.write("#### Basic Portfolio Simulation (Example using first two numeric columns as asset prices)")
                    # Simple daily returns calculation
                    asset1 = df[numeric_cols[0]].pct_change().dropna()
                    asset2 = df[numeric_cols[1]].pct_change().dropna()

                    if not asset1.empty and not asset2.empty:
                        # For simplicity, assume equal weighting
                        portfolio_returns = (asset1 + asset2) / 2
                        st.write(f"Average Daily Return (Asset 1): {asset1.mean():.4f}")
                        st.write(f"Average Daily Return (Asset 2): {asset2.mean():.4f}")
                        st.write(f"Average Daily Return (Portfolio): {portfolio_returns.mean():.4f}")
                        st.write(f"Portfolio Volatility (Std Dev): {portfolio_returns.std():.4f}")

                        fig_portfolio = px.line(portfolio_returns.cumsum().apply(np.exp), title="Cumulative Portfolio Returns (Simulated)")
                        st.plotly_chart(fig_portfolio, use_container_width=True)
                    else:
                        st.warning("Not enough data to calculate returns for selected numeric columns.")
                else:
                    st.info("Please ensure your data has at least two numeric columns for portfolio simulation.")


            st.markdown("### ⏰ Time Series Forecasting for Financial Data")
            st.info("Apply time series models to forecast financial metrics like stock prices, exchange rates, or sales.")
            st.markdown("- *You can use the **Time Series Forecasting** option in the **Model Builder** tab for this functionality.*")
            if st.button("Go to Model Builder for Time Series"):
                st.session_state.page = "🤖 Model Builder" # Attempt to navigate
                st.rerun()

            st.markdown("### 🎲 Monte Carlo Simulations")
            st.info("Run Monte Carlo simulations for financial modeling, option pricing, and risk assessment.")
            st.markdown("- *This would involve generating random paths for financial variables and calculating outcomes.*")
            if st.button("Run Simple Monte Carlo Simulation (Example)"):
                st.write("#### Simulating 1000 paths for a stock price over 252 days")
                initial_price = 100
                mu = 0.0005 # daily mean return
                sigma = 0.01 # daily standard deviation
                n_simulations = 1000
                n_days = 252

                price_paths = np.zeros((n_days, n_simulations))
                price_paths[0] = initial_price

                for t in range(1, n_days):
                    rand = np.random.normal(0, 1, n_simulations)
                    price_paths[t] = price_paths[t-1] * np.exp(mu - 0.5 * sigma**2 + sigma * rand)
                
                fig_mc = go.Figure()
                for i in range(min(10, n_simulations)): # Plot up to 10 paths
                    fig_mc.add_trace(go.Scatter(y=price_paths[:, i], mode='lines', name=f'Path {i+1}', showlegend=(i==0)))
                fig_mc.update_layout(title='Monte Carlo Simulation of Stock Price Paths',
                                     xaxis_title='Days', yaxis_title='Price')
                st.plotly_chart(fig_mc, use_container_width=True)

            st.markdown("### 💲 Financial Metrics Calculation")
            st.info("Calculate common financial metrics from your data.")
            if st.button("Calculate Basic Financial Metrics"):
                if len(numeric_cols) > 0:
                    st.write("#### Basic Financial Metrics (using first numeric column as 'Price')")
                    price_col = numeric_cols[0]
                    if not df[price_col].empty:
                        # Daily Returns
                        daily_returns = df[price_col].pct_change().dropna()
                        st.write(f"**Average Daily Returns for '{price_col}':** {daily_returns.mean():.4f}")
                        st.write(f"**Standard Deviation of Daily Returns (Volatility) for '{price_col}':** {daily_returns.std():.4f}")

                        # Simple Moving Average (e.g., 20-day SMA)
                        if len(df) > 20:
                            sma_20 = df[price_col].rolling(window=20).mean().iloc[-1]
                            st.write(f"**20-Day Simple Moving Average for '{price_col}':** {sma_20:.2f}")
                        else:
                            st.info("Not enough data points to calculate 20-day Simple Moving Average.")
                    else:
                        st.warning(f"Column '{price_col}' is empty. Cannot calculate financial metrics.")
                else:
                    st.info("Please upload data with numeric columns to calculate financial metrics.")
        else:
            st.warning("⚠️ Please upload data first to explore Financial Analytics features!")

    # ENHANCED ANALYTICS & VISUALIZATION PAGE
    elif page == "✨ Enhanced Analytics & Visualization":
        st.markdown('<h2 class="section-header">✨ Enhanced Analytics & Visualization</h2>', unsafe_allow_html=True)

        if st.session_state.data is not None:
            df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
            numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)

            st.markdown("### 🔬 Advanced Statistical Methods")
            st.write("Explore advanced statistical techniques for deeper data insights.")

            st.subheader("Bayesian Analysis and A/B Testing Framework")
            st.info("Conduct Bayesian inference and design A/B tests to make data-driven decisions.")
            st.markdown("- *This would involve libraries like `PyMC3` or `ArviZ` for Bayesian methods, and custom functions for A/B test analysis.*")
            if st.button("Simulate A/B Test Results"):
                st.write("#### Sample A/B Test Results (Conceptual)")
                # Simulate conversion rates for two groups
                np.random.seed(42)
                control_conversions = np.random.binomial(n=1000, p=0.10, size=1)
                treatment_conversions = np.random.binomial(n=1000, p=0.12, size=1)
                
                st.write(f"Control Group Conversions (out of 1000): {control_conversions[0]}")
                st.write(f"Treatment Group Conversions (out of 1000): {treatment_conversions[0]}")
                st.info("In a real scenario, statistical tests (e.g., chi-squared, t-test) or Bayesian methods would be applied here to determine significance.")

            st.subheader("Survival Analysis (Kaplan-Meier, Cox Regression)")
            st.info("Analyze time-to-event data, such as customer churn or equipment failure.")
            st.markdown("- *This would require specialized libraries like `lifelines`.*")
            if st.button("Show Kaplan-Meier Curve (Conceptual)"):
                st.write("#### Conceptual Kaplan-Meier Survival Curve")
                st.info("A Kaplan-Meier curve would show the probability of an event (e.g., churn) occurring over time.")
                # Placeholder for a plot
                fig_km = go.Figure(data=go.Scatter(x=[0, 1, 2, 3, 4, 5], y=[1.0, 0.9, 0.7, 0.5, 0.3, 0.1], mode='lines+markers'))
                fig_km.update_layout(title='Conceptual Kaplan-Meier Survival Curve', xaxis_title='Time', yaxis_title='Survival Probability')
                st.plotly_chart(fig_km, use_container_width=True)


            st.subheader("Mixed-Effects Models and Hierarchical Modeling")
            st.info("Model data with hierarchical or grouped structures, accounting for dependencies within groups.")
            st.markdown("- *This is suitable for data with repeated measures or nested structures, typically using `statsmodels` or `PyMC3`.*")

            st.subheader("Advanced Hypothesis Testing Suite (ANOVA, Non-parametric Tests)")
            st.info("Access a broader range of statistical tests for various data distributions and research questions.")
            st.markdown("- *This would expand on the existing T-test with options for ANOVA, Mann-Whitney U, Kruskal-Wallis, etc.*")

            st.subheader("Causal Inference Methods (Propensity Scoring, Instrumental Variables)")
            st.info("Estimate causal effects from observational data, addressing confounding factors.")
            st.markdown("- *This involves sophisticated statistical techniques to infer causality rather than just correlation.*")

            st.markdown("---")
            st.markdown("### 🎨 Interactive Visualizations")
            st.write("Enhance your data storytelling with advanced and dynamic visualization techniques.")

            st.subheader("Real-time Dashboards with Auto-refresh Capabilities")
            st.info("Create dashboards that automatically update with new data, providing live insights.")
            st.markdown("- *This would require a data source that updates frequently and Streamlit's `st.experimental_rerun` or `st.empty` with periodic updates.*")
            if st.button("Simulate Real-time Dashboard"):
                st.write("#### Simulated Real-time Metric")
                # Simulate a real-time metric
                current_value = np.random.rand() * 100
                st.metric("Live Metric Value", f"{current_value:.2f}")
                st.info("This metric would refresh periodically in a real-time setup.")


            st.subheader("3D Visualizations and Network Graphs")
            st.info("Visualize complex relationships and multi-dimensional data in 3D or as network structures.")
            st.markdown("- *3D plots can be created with Plotly Express, while network graphs might require `NetworkX` and `Plotly` or `Pyvis`.*")
            if st.button("Show Sample 3D Scatter Plot"):
                if len(numeric_cols) >= 3:
                    st.write("#### Sample 3D Scatter Plot (using first three numeric columns)")
                    fig_3d = px.scatter_3d(df, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2],
                                           title=f"3D Scatter: {numeric_cols[0]} vs {numeric_cols[1]} vs {numeric_cols[2]}")
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.warning("Need at least three numeric columns for a 3D scatter plot.")

            st.subheader("Geospatial Analysis with Advanced Mapping (Clustering, Heat Maps)")
            st.info("Perform location-based analysis and visualize data on interactive maps with clustering and heat map overlays.")
            st.markdown("- *This would leverage Plotly's `px.scatter_mapbox` or `px.density_mapbox` with appropriate location data (latitude, longitude).*")
            if st.button("Show Sample Geospatial Heatmap (Conceptual)"):
                st.write("#### Conceptual Geospatial Heatmap")
                st.info("This would show density of events or values on a map.")
                # Placeholder for a map
                fig_map = go.Figure(go.Scattermapbox(
                    lat=['34.0522', '37.7749', '40.7128'],
                    lon=['-118.2437', '-122.4194', '-74.0060'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(size=10, color='red'),
                    text=['LA', 'SF', 'NYC']
                ))
                fig_map.update_layout(
                    mapbox_style="open-street-map",
                    mapbox_zoom=1,
                    mapbox_center={"lat": 30, "lon": -90},
                    title="Conceptual Geospatial Map"
                )
                st.plotly_chart(fig_map, use_container_width=True)


            st.subheader("Custom Dashboard Templates and Themes")
            st.info("Apply custom visual templates and themes to your dashboards for personalized branding and aesthetics.")
            st.markdown("- *This would involve more extensive CSS styling and potentially Streamlit theming options.*")

            st.subheader("Animated Charts for Temporal Data")
            st.info("Create animated visualizations that show how data changes over time, providing dynamic insights.")
            st.markdown("- *Plotly Express supports animation frames for temporal data, often requiring a datetime column.*")
            if st.button("Show Sample Animated Chart (Conceptual)"):
                if datetime_cols and numeric_cols:
                    st.write("#### Sample Animated Chart (Conceptual: First numeric column over time)")
                    # Create a dummy animated chart
                    dummy_df = pd.DataFrame({
                        'Date': pd.to_datetime(pd.date_range(start='2020-01-01', periods=10)),
                        'Value': np.random.randint(10, 100, 10),
                        'Category': ['A', 'B'] * 5
                    })
                    fig_anim = px.line(dummy_df, x="Date", y="Value", color="Category", animation_frame="Date",
                                       title="Conceptual Animated Line Chart")
                    st.plotly_chart(fig_anim, use_container_width=True)
                else:
                    st.warning("Need both datetime and numeric columns for animated charts.")

        else:
            st.warning("⚠️ Please upload data first to explore Enhanced Analytics & Visualization features!")


def main_app():
    """Main application content (protected area)"""
    
    # Sidebar with user info and logout
    with st.sidebar:
        st.header("User Menu")
        st.write(f"Logged in as: **{st.session_state.username}**")
        
        if st.button("👤 Profile", use_container_width=True):
            st.session_state.show_profile = True
        
        if st.button("🚪 Logout", type="primary", use_container_width=True):
            logout()
    
    # Show profile if requested, otherwise show DataScope app
    if st.session_state.get('show_profile'):
        user_profile()
    else:
        datascope_app_content() # This calls the main DataScope app logic

# Main application flow
def main():
    """Main application entry point"""
    # Configure page - MUST BE THE FIRST STREAMLIT COMMAND
    st.set_page_config(
        page_title="DataScope Analytics App",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="auto"
    )

    # Initialize database and session state
    init_database()
    init_session_state() # This now initializes all session state variables
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
    }
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #f8f9fa;
    }
    .section-header {
        color: #667eea;
        border-bottom: 2px solid #764ba2;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Authentication logic
    if st.session_state.authentication_status is True:
        # User is authenticated - show main app content
        main_app()
    
    elif st.session_state.authentication_status is False:
        # Authentication failed
        st.error("❌ Authentication failed")
        
        # Show login and registration options
        st.markdown('<div class="main-header"><h1>🔐 Secure Application</h1><p>Please log in to continue</p></div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            with st.container():
                login_form()
        
        with tab2:
            with st.container():
                registration_form()
        
        # Additional info
        with st.expander("ℹ️ Password Requirements"):
            st.write("""
            Your password must contain:
            - At least 8 characters
            - At least one uppercase letter
            - At least one lowercase letter
            - At least one number
            - At least one special character (!@#$%^&*(),.?\":{}|<>)
            """)
    
    else:
        # Not authenticated yet (initial load or after logout) - show login/register options
        st.markdown('<div class="main-header"><h1>🔐 Secure Application</h1><p>Please log in or create an account to continue</p></div>', unsafe_allow_html=True)
        
        # Create tabs for login and registration
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            with st.container():
                login_form()
        
        with tab2:
            with st.container():
                registration_form()
        
        # Additional info
        with st.expander("ℹ️ Password Requirements"):
            st.write("""
            Your password must contain:
            - At least 8 characters
            - At least one uppercase letter
            - At least one lowercase letter
            - At least one number
            - At least one special character (!@#$%^&*(),.?\":{}|<>)
            """)

if __name__ == "__main__":
    main()
