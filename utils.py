import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import io
import re


MAX_CATEGORICAL_FEATURES_FOR_OHE = 10

PLOTLY_TEMPLATE = "plotly_dark"
def generate_data_summary(df: pd.DataFrame) -> str:
    """
    Generates a comprehensive summary of the DataFrame.

    Args:
        df: The input Pandas DataFrame.

    Returns:
        A string containing a multi-faceted summary of the DataFrame,
        including shape, data types, head, descriptive statistics,
        missing values, memory usage, and df.info().
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    summary = f"""
### 📋 Full Data Summary:

1. Shape of the Dataset:
   - Rows: {df.shape[0]}
   - Columns: {df.shape[1]}

2. Data Types:

{df.dtypes.to_string()}


3. First 5 Rows (Head):

{df.head().to_markdown(index=False)}


4. Descriptive Statistics (for numerical columns):

{df.describe().to_markdown()}


5. Missing Values (sum per column):
"""
    missing_values = df.isnull().sum()
    missing_values_str = missing_values[missing_values > 0].to_string()
    if not missing_values_str.strip() or missing_values[missing_values > 0].empty:
        summary += "   - ✅ No missing values found.\n"
    else:
        summary += f"\n{missing_values_str}\n\n"

    summary += f"""
6. Memory Usage:
   - {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB

7. Column Information (from df.info()):

{info_str}

"""
    return summary.strip()

def extract_column_names(text: str, df_columns: list, max_cols: int = 1) -> list:
    """
    Extracts one or more column names from a given text string,
    prioritizing quoted names and then direct word matches against DataFrame columns.

    Args:
        text: The user input string.
        df_columns: A list of column names in the DataFrame.
        max_cols: The maximum number of column names to extract.

    Returns:
        A list of extracted column names (original casing) up to max_cols.
    """
    text_lower = text.lower()
    original_df_columns_map = {col.lower(): col for col in df_columns}
    df_columns_lower = list(original_df_columns_map.keys())

    extracted_cols = []


    quoted_cols_match = re.findall(r"'(.?)'|\"(.?)\"", text_lower)
    for q_col_tuple in quoted_cols_match:
        q_col = next((s for s in q_col_tuple if s), None)
        if q_col and q_col in df_columns_lower:
            original_col_name = original_df_columns_map[q_col]
            if original_col_name not in extracted_cols: 
                extracted_cols.append(original_col_name)
                if len(extracted_cols) == max_cols:
                    return extracted_cols


    words_in_text = set(re.findall(r'\b\w+\b', text_lower)) 

    for col_l in df_columns_lower:
        original_col_name = original_df_columns_map[col_l]
        if original_col_name in extracted_cols: 
            continue


        if col_l in text_lower:
            extracted_cols.append(original_col_name)
            if len(extracted_cols) == max_cols:
                return extracted_cols
            continue 


        col_name_parts = set(col_l.split())
        if words_in_text.intersection(col_name_parts):
            extracted_cols.append(original_col_name)
            if len(extracted_cols) == max_cols:
                return extracted_cols

    return extracted_cols


def get_column_stats(user_input: str, df: pd.DataFrame) -> str:
    """
    Calculates and returns descriptive statistics for a specified column.
    Differentiates between numerical and categorical columns for relevant stats.

    Args:
        user_input: The user's query string.
        df: The input Pandas DataFrame.

    Returns:
        A string containing formatted statistics for the identified column,
        or an error message if the column is not found or not identifiable.
    """
    cols_found = extract_column_names(user_input, df.columns, max_cols=1)
    if not cols_found:
        return "❌ Could not identify a valid column name in your request. Please specify clearly, e.g., 'stats for column_name' or 'mean of Age'."
    col = cols_found[0]

    if col not in df.columns:
         return f"❌ Column '{col}' not found in the dataset."

    column_data = df[col]

    if pd.api.types.is_numeric_dtype(column_data):
        desc = column_data.describe()
        mode_series = column_data.mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else "N/A (no distinct mode)"
        skewness = column_data.skew()
        kurt = column_data.kurtosis()

        return f"""📊 Statistics for Numerical Column: {col}**
- Count:    {desc.get('count', 0):.0f}
- Mean:     {desc.get('mean', float('nan')):.2f}
- Median:   {desc.get('50%', float('nan')):.2f} (50th percentile)
- Mode:     {mode_val}
- Std Dev:  {desc.get('std', float('nan')):.2f}
- Min:      {desc.get('min', float('nan')):.2f}
- Max:      {desc.get('max', float('nan')):.2f}
- 25th Pctl:{desc.get('25%', float('nan')):.2f}
- 75th Pctl:{desc.get('75%', float('nan')):.2f}
- Skewness: {skewness:.2f}
- Kurtosis: {kurt:.2f}
"""
    else: 
        desc = column_data.describe() 
        unique_vals = column_data.nunique()
        mode_series = column_data.mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else "N/A"

        return f"""📊 Statistics for Categorical/Object Column: {col}**
- Count:        {desc.get('count', 0)}
- Unique Values:{unique_vals}
- Top (Most Frequent): {desc.get('top', 'N/A')}
- Frequency of Top: {desc.get('freq', 'N/A')}
- Mode (Calculated): {mode_val}
"""

def generate_plot_interactive(user_input: str, df: pd.DataFrame) -> dict | str:
    """
    Generates an interactive Plotly chart based on user input.
    Supports histograms, box plots, scatter plots, and correlation heatmaps.

    Args:
        user_input: The user's query string specifying the plot type and columns.
        df: The input Pandas DataFrame.

    Returns:
        A dictionary containing the Plotly figure and a message if successful,
        or a string message indicating an error or need for clarification.
        The dictionary format is: {"type": "plotly", "fig": fig_object, "message": "success_message"}
    """
    input_lower = user_input.lower()
    fig = None
    message = ""

    try:
        if "histogram" in input_lower or "distribution of" in input_lower:
            cols = extract_column_names(input_lower, df.columns, 1)
            if cols:
                col_x = cols[0]
                if pd.api.types.is_numeric_dtype(df[col_x]):
                    fig = px.histogram(df, x=col_x, title=f"Histogram of {col_x}", marginal="box", template=PLOTLY_TEMPLATE)
                    message = f"✅ Displaying histogram for numerical column {col_x}."
                else:
                    fig = px.histogram(df, x=col_x, title=f"Bar Chart (Counts) of {col_x}", template=PLOTLY_TEMPLATE)
                    fig.update_xaxes(categoryorder="total descending")
                    message = f"✅ Displaying bar chart (counts) for categorical column {col_x}."
            else:
                return "❓ For a histogram or bar chart, please specify a column. E.g., 'plot histogram of age'."

        elif "box plot" in input_lower or "boxplot of" in input_lower:
            cols = extract_column_names(input_lower, df.columns, 1) 
            if cols:
                col_y = cols[0]
                if pd.api.types.is_numeric_dtype(df[col_y]):
                    fig = px.box(df, y=col_y, title=f"Box Plot of {col_y}", template=PLOTLY_TEMPLATE)
                    message = f"✅ Displaying box plot for {col_y}."
                else:
                    return f"⚠ Box plots are best for numerical data. {col_y} is not numeric. Try a bar chart."
            else:
                return "❓ For a box plot, please specify a numerical column. E.g., 'show box plot for salary'."

        elif "scatter plot" in input_lower or "scatter of" in input_lower:
            cols = extract_column_names(input_lower, df.columns, 2) 
            if len(cols) == 2:
                col_x, col_y = cols[0], cols[1]
                if pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(df[col_y]):
                    fig = px.scatter(df, x=col_x, y=col_y, title=f"Scatter Plot: {col_x} vs {col_y}",
                                     trendline="ols", trendline_color_override="red", template=PLOTLY_TEMPLATE)
                    message = f"✅ Displaying scatter plot for {col_x} vs {col_y} with a trendline."
                else:

                    return f"⚠ Scatter plots typically require two numerical columns. Please check the data types for {col_x} and {col_y}."
            else:
                return "❓ For a scatter plot, please specify two columns. E.g., 'scatter plot of age vs salary'."

        elif "correlation heatmap" in input_lower or "correlation matrix plot" in input_lower:
             return get_correlation_matrix(df) 

        else: 
            cols = extract_column_names(input_lower, df.columns, 1)
            if cols:
                col_to_plot = cols[0]
                if pd.api.types.is_numeric_dtype(df[col_to_plot]):
                    fig = px.histogram(df, x=col_to_plot, title=f"Distribution of {col_to_plot}", marginal="box", template=PLOTLY_TEMPLATE)
                    message = f"✅ Defaulted to histogram for numerical column {col_to_plot} as no specific plot type was mentioned."
                else:

                    counts = df[col_to_plot].value_counts().reset_index()
                    counts.columns = [col_to_plot, 'count'] 
                    fig = px.bar(counts, x=col_to_plot, y='count', title=f"Value Counts for {col_to_plot}", template=PLOTLY_TEMPLATE)
                    fig.update_xaxes(categoryorder="total descending")
                    message = f"✅ Defaulted to bar chart for categorical column {col_to_plot} as no specific plot type was mentioned."
            else:
                return "❓ Please specify what you want to plot and for which column(s). E.g., 'plot histogram of age' or 'scatter plot of salary vs experience'."

        if fig:
            return {"type": "plotly", "fig": fig, "message": message}
        else:
            return message if message else "❓ Could not determine the plot type or column. Please be more specific."

    except Exception as e:
        return f"❌ Error generating plot: {str(e)}"


def get_correlation_matrix(df: pd.DataFrame) -> dict | str:
    """
    Computes and returns a correlation matrix heatmap for numerical columns.

    Args:
        df: The input Pandas DataFrame.

    Returns:
        A dictionary containing the Plotly figure for the heatmap and a message,
        or a string message if not enough numerical columns are available.
    """
    numerical_df = df.select_dtypes(include=np.number)
    if numerical_df.shape[1] < 2:
        return "⚠ Not enough numerical columns to compute a correlation matrix (at least 2 are required)."

    corr_matrix = numerical_df.corr()
    fig = px.imshow(corr_matrix, text_auto=".2f", 
                    aspect="auto", 
                    title="Correlation Heatmap of Numerical Features",
                    color_continuous_scale='RdBu_r', 
                    template=PLOTLY_TEMPLATE)
    return {"type": "plotly", "fig": fig, "message": "✅ Correlation heatmap displayed."}


def extract_target_column(user_input: str, df_columns: pd.Index) -> str | None:
    """
    Extracts the target column for machine learning tasks from user input.
    It looks for explicit mentions like "predict 'column_name'" or "target is 'column_name'",
    or infers from context. Falls back to the last column if no clear target is found.

    Args:
        user_input: The user's query string.
        df_columns: A Pandas Index object containing the column names of the DataFrame.

    Returns:
        The identified target column name (original casing), or the last column name as a fallback.
        Returns None if df_columns is empty.
    """
    if df_columns.empty: 
        return None

    user_input_lower = user_input.lower()
    df_columns_lower_map = {col.lower(): col for col in df_columns}

 
    patterns = [
        r"predict\s+(?:column\s+)?(['\"]?(.*?)['\"]?)\b",
        r"target\s*(?:is|=)\s*(?:column\s+)?(['\"]?(.*?)['\"]?)\b",
        r"for\s+column\s+(['\"]?(.*?)['\"]?)\b",
        r"dependent\s*(?:variable\s*(?:is|=)?)?\s*(['\"]?(.*?)['\"]?)\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input_lower)
        if match:

            potential_target_in_match = next((g.strip() for g in match.groups() if g and g.strip()), None)
            if potential_target_in_match and potential_target_in_match in df_columns_lower_map:
                return df_columns_lower_map[potential_target_in_match]


    cols_from_general_extraction = extract_column_names(user_input_lower, list(df_columns), 1) # Convert Index to list
    if cols_from_general_extraction:
        return cols_from_general_extraction[0]

    return df_columns[-1]


def train_and_evaluate_model(user_input: str, df: pd.DataFrame) -> str:
    """
    Trains a simple machine learning model (Regression or Classification)
    based on the user input and DataFrame. Includes basic preprocessing,
    model training, evaluation, and feature importance (if applicable).

    Args:
        user_input: The user's query, used to infer the target variable.
        df: The input Pandas DataFrame.

    Returns:
        A string containing the model training results and evaluation metrics,
        or an error message if training fails.
    """
    try:
        target_col = extract_target_column(user_input, df.columns)
        if not target_col:
            return "❌ Could not determine the target variable for model training. Please specify it, e.g., 'train model to predict ColumnName'."

        df_processed = df.copy()
        df_processed.dropna(subset=[target_col], inplace=True) 

        if df_processed.empty:
            return f"❌ Dataset became empty after removing rows with missing target values in '{target_col}'. Cannot train model."

        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        high_cardinality_cats = []
        categorical_features_for_ohe = []
        for col in categorical_features:
            if X[col].nunique() > MAX_CATEGORICAL_FEATURES_FOR_OHE:
                high_cardinality_cats.append(col)
            else:
                categorical_features_for_ohe.append(col)
        

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features_for_ohe)
            ],
            remainder='drop'


        label_encoder = None 
        if pd.api.types.is_numeric_dtype(y):
            if y.nunique() > 15 and y.nunique() / len(y) > 0.1:
                model_type = 'regression'
                model = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1, min_samples_leaf=5)
                scoring_metric_name = 'R² Score'
                secondary_metric_name = 'Mean Squared Error (MSE)'
            else:
                model_type = 'classification'
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1, min_samples_leaf=5)
                scoring_metric_name = 'Accuracy'
                secondary_metric_name = 'Classification Report'
        else: 
            model_type = 'classification'
            label_encoder = LabelEncoder() 
            y = label_encoder.fit_transform(y)
            model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1, min_samples_leaf=5)
            scoring_metric_name = 'Accuracy'
            secondary_metric_name = 'Classification Report'

        if len(X) < 10 or (model_type == 'classification' and len(np.unique(y)) < 2):
             return f"❌ Insufficient data or too few unique classes in target '{target_col}' to train a meaningful {model_type} model after preprocessing."

        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])

        if y.shape[0] < 5: 
            return "❌ Not enough samples to split into training and testing sets after initial preprocessing."

        test_size = 0.25
        stratify_param = y if model_type == 'classification' and len(np.unique(y)) > 1 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify_param)
        except ValueError as e: 
            if "The least populated class" in str(e) or "n_splits=" in str(e):
                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            else:
                raise e 

        if X_train.empty or X_test.empty:
            return "❌ Training or testing set became empty after splitting. This usually means there's very little data."

        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)

        result_str = f"### 🤖 Model Training Results for Target: {target_col}\n"
        result_str += f"- Model Type: {model._class.name_} ({model_type})\n"
        
        try:
            feature_names_out = full_pipeline.named_steps['preprocessor'].get_feature_names_out()
            result_str += f"- Number of Features Used by Model (after preprocessing): {len(feature_names_out)}\n"
        except Exception:
            result_str += "- Number of features used could not be determined post-preprocessing.\n"

        if high_cardinality_cats:
            result_str += f"- Categorical features dropped due to high cardinality (> {MAX_CATEGORICAL_FEATURES_FOR_OHE} unique values): {', '.join(high_cardinality_cats)}\n"
        if not numerical_features and not categorical_features_for_ohe:
             result_str += "- Warning: No features were selected for preprocessing. The model might not be meaningful.\n"


        if model_type == 'regression':
            score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            result_str += f"- {scoring_metric_name}: {score:.3f}\n"
            result_str += f"- {secondary_metric_name}: {mse:.3f}\n"
        else:
            score = accuracy_score(y_test, y_pred)
            target_names_for_report = [str(cls_name) for cls_name in label_encoder.classes_] if label_encoder else None
            report = classification_report(y_test, y_pred, target_names=target_names_for_report, zero_division=0)
            result_str += f"- {scoring_metric_name}: {score:.3f}\n"
            result_str += f"\n*{secondary_metric_name}:*\n\n{report}\n\n"

        if hasattr(model, 'feature_importances_'):
            try:
                feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(10) # Show top 10
                result_str += f"\n*Top Feature Importances (up to 10):*\n\n{feature_importance_df.to_string(index=False)}\n\n"
            except Exception as fi_e:
                print(f"Could not get feature importances: {fi_e}") # Log error
                result_str += "- Feature importances could not be extracted for this pipeline setup.\n"

        return result_str + "\n\n*Note: This is a simplified model trained for quick analysis. For production, more rigorous feature engineering, hyperparameter tuning, and model selection would be required.*"

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc() 
        return f"❌ Failed to train model: {str(e)}\n\n<details><summary>Click for technical traceback</summary>\n\n\n{tb_str}\n\n</details>"

def get_plot_function_and_cols(user_input: str, df: pd.DataFrame):
    """
    This function is a conceptual placeholder.
    It would parse the user input to determine the plot type and relevant columns,
    returning this information for another part of the application to use for plotting.
    Currently, generate_plot_interactive handles both parsing and figure generation.
    """

    pass
