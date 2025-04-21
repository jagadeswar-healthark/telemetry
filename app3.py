import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
from scipy.stats import linregress
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import joblib
import warnings
import altair as alt
warnings.filterwarnings(action="ignore")

# --- CSS Styling for Red and White Theme ---
st.markdown("""
    <style>
    /* Styling for the main submit button */
    div.stButton > button {
        background-color: red;  /* Red background */
        width: 200px;           /* Adjust width */
        height: 40px;           /* Adjust height */
        border: none;           /* Remove border */
        cursor: pointer;        /* Change cursor on hover */
        border-radius: 25px;    /* Rounded corners */
        color: white;           /* White text color */
        border: 2px solid white; /* Border */
        margin-top: 5px;
    }
    /* Hover effect for the buttons */
    div.stButton > button:hover {
        background-color: white;
        color: red;
        border: 2px solid red;
    }
    body { background-color: #ffffff; color: #000000; }
    .title { font-size: 28px; font-weight: bold; color: #FF0000; text-align: center; }
    .output-box { border: 2px solid #e60012; border-radius: 15px; padding: 15px; margin-top: 20px; background: #fff; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Logo and Toggle ---
# Replace with your logo path or URL
logo_url = "logo1.png"  # Update with actual logo path
st.sidebar.image(logo_url, width=200)

if 'show_menu' not in st.session_state:
    st.session_state.show_menu = False

if st.sidebar.button("Show Telemetry Menu"):
    st.session_state.show_menu = True

# Header
st.markdown("<h1 class='title'>Telemetry Analysis</h1>", unsafe_allow_html=True)

# --- Preprocessing Function ---
def preprocess_puc_file(file_content, columns, window=60):
    raw_data = file_content.decode('utf-8')
    lines = raw_data.split('\n')
    data_lines = [line for line in lines if not line.startswith('PUC_VER')]

    if not data_lines:
        return None

    expected_col_count = len(data_lines[0].split(','))
    valid_lines = [line for line in data_lines if len(line.split(',')) == expected_col_count]
    data_str = "\n".join(valid_lines)

    df = pd.read_csv(StringIO(data_str), header=None)
    df.columns = columns[:df.shape[1]]
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')

    st.write("**The uploaded file is being analyzed from**")
    st.write(f"Start Date: {df['Date/Time'].min()}")
    st.write(f"End Date: {df['Date/Time'].max()}")

    three_months_ago = df['Date/Time'].max() - pd.DateOffset(months=3)
    df_last_3_months = df[df['Date/Time'] >= three_months_ago]

    print("**Last 3 Months**")
    print(f"Start Date: {df_last_3_months['Date/Time'].min()}")
    print(f"End Date: {df_last_3_months['Date/Time'].max()}")

    df = df_last_3_months

    # Feature engineering
    df['Diff_RTD_Setpoint'] = df['RTD'] - df['Setpoint']
    df['lower_bound_RTD'] = df['Setpoint'] - 2
    df['upper_bound_RTD'] = df['Setpoint'] + 2

    df['TC1_in_range'] = df['TC1'].between(-35, -20)
    df['TC6_in_range'] = df['TC6'].between(-30, -15)
    df['TC3_in_range'] = df['TC3'].between(-95, -86)
    df['TC4_in_range'] = df['TC4'].between(-75, -40)
    df['TC10_in_range'] = df['TC10'].between(-45, -35)
    df['RTD_in_range'] = (df['RTD'] >= df['lower_bound_RTD']) & (df['RTD'] <= df['upper_bound_RTD'])

    def compute_slope(series):
        y = series.values
        x = np.arange(len(y)).reshape(-1, 1)
        if len(y) < 2 or np.all(np.isnan(y)):
            return np.nan
        model = LinearRegression().fit(x, y)
        return model.coef_[0]

    for col in ['RTD', 'TC1', 'TC3', 'TC4', 'TC6', 'TC10']:
        df[f'{col}_trend'] = df[col].rolling(window).apply(compute_slope, raw=False)

    df['PUC_state_mean'] = df['PUC State'].rolling(window).mean()
    df['Diff_RTD_Setpoint_mean'] = df['Diff_RTD_Setpoint'].rolling(window).mean()

    return df.dropna(), df_last_3_months

# --- Column Names ---
columns = [
    'Date/Time', 'RTD', 'TC1', 'TC2', 'TC3', 'TC4', 'TC6', 'TC7', 'TC9', 'TC10',
    'Setpoint', 'Line Input', 'PUC State', 'User Offset', 'Warm warning setpoint', 'Cold warning setpoint',
    'Stage1 RPM', 'Stage2 RPM', 'Stage3 RPM', 'Valve Step', 
    'Condenser Fan RPM', 'Algorithm Flags', 'Algo State', 'BUS RTD', 'S1 Pressure',
    'Superheat', 'S2 Temperature', 'TSat'
]

# --- Main App Logic ---
if st.session_state.show_menu:
    uploaded_file = st.file_uploader("Upload your Telemetry Data file", type=["puc"])
    

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        if st.button("Submit"):
            raw_data = uploaded_file.read()

            # Load model and features
            try:
                model = joblib.load("rf_failure_model_V1404.pkl")
                features = joblib.load("model_features_V1404.pkl")
                print("**Loaded Features**: ", features)
            except Exception as e:
                st.error(f"Error loading model or features: {e}")
                st.stop()

            # Preprocess data
            new_df, df_last_3_months = preprocess_puc_file(raw_data, columns)
            if new_df is None:
                st.error("No valid data found in the uploaded file.")
                st.stop()

            # --- Compute Trend-Based Failure Flags ---
            # Precompute TC3 and TC4 mean
            tc3_tc4_mean_zero = (new_df[['TC3', 'TC4']].mean(axis=1) == 0)

            # Step 2: Compute trend-based failure flags
            condition_1 = (
                (new_df['RTD_in_range'] == False) &
                (new_df['TC1_trend'] > 0) &
                (new_df['TC10_trend'] > 0) &
                (new_df['PUC_state_mean'] == 1) &
                (new_df['RTD_trend'] > 0)
            )
            condition_2 = (
                (new_df['RTD_in_range'] == False) &
                (new_df['TC3_trend'] < 0) &
                (new_df['TC4_trend'] > 0) &
                (new_df['TC10_trend'] > 0) &
                (new_df['PUC_state_mean'] == 1) &
                (new_df['RTD_trend'] > 0)
            )
            condition_3 = (
                (new_df['RTD_in_range'] == False) &
                (new_df['TC6_trend'] > 0) &
                (new_df['TC10_trend'] < 0) &
                (new_df['PUC_state_mean'] == 3) &
                (new_df['RTD_trend'] > 0)
            )
            condition_4 = (
                (new_df['RTD_in_range'] == False) &
                (new_df['TC3_trend'] < 0) &
                (new_df['TC4_trend'] > 0) &
                (new_df['TC10_trend'] < 0) &
                (new_df['PUC_state_mean'] == 3) &
                (new_df['RTD_trend'] > 0)
            )

            # Extended conditions with TC3 and TC4 mean = 0
            condition_1_old_device = condition_1 & tc3_tc4_mean_zero
            condition_3_old_device = condition_3 & tc3_tc4_mean_zero

            # Final classification using np.select
            new_df['Trend_Flag'] = np.select(
                [
                    condition_1_old_device,
                    condition_3_old_device,
                    condition_1,
                    condition_2,
                    condition_3,
                    condition_4
                ],
                [
                    "1st stage Compression failure or 1st Stage Leakage Issue",
                    "2nd stage Compression failure or 2nd Stage Leakage Issue",
                    '1st stage compression failure',
                    '1st stage leakage failure',
                    '2nd stage compression failure',
                    '2nd stage leakage failure'
                ],
                default='No issue detected - your device is working properly'
            )

            # --- Flag Sustained Sequences ---
            def flag_sustained(df, col='Trend_Flag', min_consecutive=3):
                sustained_flags = [False] * len(df)
                count = 0
                for i, val in enumerate(df[col]):
                    if val != 'No issue detected - your device is working properly':
                        count += 1
                        if count >= min_consecutive:
                            for j in range(i - count + 1, i + 1):
                                sustained_flags[j] = True
                    else:
                        count = 0
                return sustained_flags

            new_df['Sustained_Issue'] = flag_sustained(new_df)
            new_df['Issue_Detected'] = new_df['Sustained_Issue'].astype(int)

            flagged = new_df[new_df['Sustained_Issue']]
            if not flagged.empty:
                print("Flagged Data Points")
                print(flagged[['Date/Time', 'TC1', 'TC10', 'TC3', 'TC6', 'TC4', 'RTD', 'RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend', 'PUC State']])

            total_rows = len(new_df)
            total_issues = new_df['Issue_Detected'].sum()

            # --- Model Predictions ---
            X_new = new_df[features]
            predictions = model.predict(X_new)

            label_mapping = {
                0: "No issue detected - your device is working properly",
                1: '1st stage Compression failure or 1st Stage Leakage Issue',
                2: "2nd stage Compression failure or 2nd Stage Leakage Issue",
                3: '1st stage compression failure',
                4: "1st stage leakage failure",
                5: "2nd stage compression failure",
                6: "2nd Stage leakage failure"
            }
            new_df['Prediction_Code'] = predictions
            new_df['Prediction_Label'] = [label_mapping.get(p, "Unknown") for p in predictions]
            
            new_df['Final_Label'] = new_df.apply(lambda row: row['Trend_Flag'] if row['Sustained_Issue'] else row['Prediction_Label'], axis=1)

            count_label = new_df['Final_Label'].value_counts()
            unique_count = new_df['Final_Label'].unique()
            # Define suggestions for each label
            suggestions = {
                        "1st stage compression failure": (
                            "Suggest to verify the flash sequence of the 1st stage to confirm any issue with the compressor"
                        ),
                        "1st stage leakage failure": (
                            "Suggest to leak test the 1st stage to rule out any leak. Especially around condenser inler and outlet braze connections. Request them to report back the leak location to US."
                        ),
                        "1st stage Compression failure or 1st Stage Leakage Issue": ("Recommend checking the flash sequence of the 1st stage to confirm any compressor-related issues" or "Recommend performing a leak test on the 1st stage to eliminate the possibility of leakage, particularly near the condenser inlet and outlet braze joints. Request them to report back the leak location to US."),
                        "2nd stage compression failure": (
                            "suggesting an overcompensation or imbalance in the system. "
                            "Additional Suggestions: Analyze historical temperature trends to identify patterns that may indicate potential compressor issues before they escalate. "
                            "Perform a detailed inspection of the 2nd stage compressor wiring and connections to rule out electrical faults contributing to the failure. "
                            "Conduct a pressure test on the 2nd stage system to detect any underlying issues that might affect the flash sequence or compressor performance."
                        ),
                        "2nd stage leakage failure": (
                            "suggesting an overcompensation or imbalance in the system."
                        ),
                        "2nd stage Compression failure or 2nd Stage Leakage Issue":(
                        "Indicating a possible overcompensation or system imbalance."
                        "Further Recommendations: Review historical temperature data to uncover patterns that could signal early-stage compressor problems."
                                                "Inspect the wiring and connections of the 2nd stage compressor thoroughly to eliminate any potential electrical faults."
                                                "Carry out a pressure test on the 2nd stage system to uncover any hidden issues that may impact the flash sequence or the overall performance of the compressor."),
                        "No issue detected - your device is working properly": (
                            "No issue detected - your device is working properly. No action required."
                        ),
                    }

            # Count the occurrences of each label
            count_label = new_df['Final_Label'].value_counts()
            unique_count = new_df['Final_Label'].unique()

            # Determine the result
            if len(unique_count) == 1 and 'No issue detected - your device is working properly' in unique_count:
                result = "No issue detected - your device is working properly"
            else:
                # Exclude 'No issue detected' and find the label with the maximum count
                failure_labels = count_label.drop(labels=['No issue detected - your device is working properly'], errors='ignore')
                result = failure_labels.idxmax()
                flagged_start = flagged['Date/Time'].min()
                flagged_end = flagged['Date/Time'].max()
                print("Flagged Start: ", flagged_start)
                print("Flagged End: ", flagged_end)
                if pd.isna(flagged_start) or pd.isna(flagged_end):
                    result = "No issue detected - your device is working properly"
                    print("No issue detected - your device is working properly")
                else:
                    print("checking: ", result)
                    result = result

            # Get the suggestion based on the result
            suggestion = suggestions.get(result, "Unknown issue detected. Please investigate further.")

            # Output the result and suggestion
            print(f"Result: {result}")
            print(f"Suggestion: {suggestion}")
            # --- Evaluation Metrics ---
            y_true = new_df['Trend_Flag']
            y_pred = new_df['Final_Label']
            accuracy = accuracy_score(y_true, y_pred)
            print("Model Performance")
            print(f"**Accuracy**: {accuracy:.2%}")
            accuracy = str(accuracy)
            accuracy = accuracy[:4]
            print("**Classification Report**:")
            print(classification_report(y_true, y_pred))

            df = new_df

            # --- Summary Statistics ---
            core_columns = ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']
            trend_columns = [col + '_trend' for col in core_columns]
            columns_to_check = core_columns + trend_columns

            # Prepare summary
            summary = []

            # Check if 'flagged' exists and is not empty
            if 'flagged' in locals() and not flagged.empty:
                data_source = flagged
                source_label = "Flagged Period"
            else:
                data_source = df  # fallback to full data
                source_label = "Full Dataset (No Issues Detected)"

            # Loop through each column and collect stats
            for col in columns_to_check:
                if col in data_source.columns:
                    min_val = data_source[col].min()
                    max_val = data_source[col].max()
                    mean_val = data_source[col].mean()
                    min_date = data_source.loc[data_source[col].idxmin(), 'Date/Time']
                    max_date = data_source.loc[data_source[col].idxmax(), 'Date/Time']

                    summary.append({
                        'Column': col,
                        'Min': min_val,
                        'Min Date': min_date,
                        'Mean': mean_val,
                        'Max': max_val,
                        'Max Date': max_date
                    })

            summary_df = pd.DataFrame(summary)
            print(f"📋 Summary Statistics ({source_label})")
            st.write(summary_df)

            tc1_trend = 'Decreasing' if flagged['TC1_trend'].mean() < 0 else 'Increasing' if not flagged.empty else 'Stable'
            tc10_trend = 'Decreasing' if flagged['TC10_trend'].mean() < 0 else 'Increasing' if not flagged.empty else 'Stable'
            rtd_mean = flagged['RTD'].mean() if not flagged.empty else df['RTD'].mean()
            setpoint_mean = flagged['Setpoint'].mean() if not flagged.empty else df['Setpoint'].mean()
            rtd_vs_setpoint_status = "RTD higher than Setpoint consistently" if rtd_mean > setpoint_mean else "RTD within expected range"
            root_cause = flagged['Trend_Flag'].value_counts().idxmax() if not flagged.empty and 'Trend_Flag' in flagged.columns else "No root cause detected"
            device_status = "Issue Detected" if total_issues > 0 else "Working Well"
            print(root_cause)
            state_puc = {
                        "1st stage compression failure": (
                            "1"
                        ),
                        "1st stage leakage failure": (
                            "1"
                        ),
                        "1st stage Compression failure or 1st Stage Leakage Issue": ("1"),
                        "2nd stage compression failure": (
                            "3"
                        ),
                        "2nd stage leakage failure": (
                            "3"
                        ),
                        "2nd stage Compression failure or 2nd Stage Leakage Issue":(
                        "3"),
                    }

            pucState = state_puc.get(root_cause)
            print(pucState)
            if root_cause == "No root cause detected":
                summary_sugg_var = f"""
            #### 🧠 Root Cause Explanation:
            - No root cause detected.

            **Suggested Preventive Actions:**
            - No action required.

            🛠 **Suggested Corrective Actions:**
            - No action required.

            ✅ **Confidence Level in Root Cause Identification:** **{accuracy}**

            (Note: accuracy ranges from 0 to 1, with 1 being the most accurate prediction.)
            """
            elif root_cause in state_puc:
                pucState = state_puc.get(root_cause)
                print("PUC State: ", pucState)
                summary_sugg_var = f"""
            #### 🧠 Root Cause Explanation:
            - The combination of a **{tc1_trend.lower()}** 1st Suction line and **{tc10_trend.lower()}** Heat exchange (BPHX), while the system remains in state {pucState}, suggests abnormal heat transfer or inefficiencies likely due to a **{root_cause}**. Persistent RTD elevation beyond the setpoint supports the hypothesis of system load imbalance or cooling inefficiency.

            **🔧 Suggested Preventive Actions:**
            - {suggestion}
            - Schedule periodic pressure integrity tests for both compressor stages.
            - Regularly inspect thermal coupling and ensure adequate insulation around 1st Suction line/Heat exchange (BPHX) lines.
            - Implement alerts when TC trends diverge while PUC remains constant.

            🛠 **Suggested Corrective Actions:**
            - Conduct a diagnostic check on the suspected compressor module.
            - Inspect and replace any worn or damaged seals that could cause internal leaks.
            - Recalibrate sensors and verify PID control settings for thermal regulation.

            ✅ **Confidence Level in Root Cause Identification:** **{accuracy}**

            (Note: accuracy ranges from 0 to 1, with 1 being the most accurate prediction.)
            """

            st.markdown(f"""
            <div class='output-box'>

            #### 📊 GenAI Summary: Telemetry-Based Preventive Maintenance Analysis

            **Observation:**
                        
            Uploaded file is analyzed from {df_last_3_months['Date/Time'].min()} to {df_last_3_months['Date/Time'].max()}.
            A total of **{total_rows}** telemetry time points were analyzed from above Date Range.  
            The system detected **{total_issues}** potential issue(s) where:
            - 1st Suction line was **{tc1_trend.lower()}**
            - Heat exchange (BPHX) was **{tc10_trend.lower()}**
            - PUC State remained in condition {pucState}.

            A Random Forest Classifier trained on trend features achieved an accuracy of **{accuracy}**.  
            Feature importance indicates that **1st Suction line Trend** and **Heat exchange(BPHX) Trend** are strong indicators of issue detection.

            ---

            #### 🔎 Technical Evaluation Summary

            **Device Status:** {device_status}  
            **Detected Root Cause:** **{root_cause}**

            **Key Sensor Readings & Trends:**
            - 1st Suction line Trend: **{tc1_trend}**
            - Heat exchange(BPHX) Trend: **{tc10_trend}**
            - RTD vs Setpoint: **{rtd_vs_setpoint_status}**

            ---

            {summary_sugg_var}
            """, unsafe_allow_html=True)

            # Define a consistent figure size for all plots
            st.markdown("### Visualizations", unsafe_allow_html=True)

            # Ensure Date/Time is in datetime format
            df['Date/Time'] = pd.to_datetime(df['Date/Time'])

            # Determine plot time range
            if 'flagged' in locals() and not flagged.empty:
                flagged['Date/Time'] = pd.to_datetime(flagged['Date/Time'])
                flagged_start = flagged['Date/Time'].min()
                flagged_end = flagged['Date/Time'].max()
                plot_start = flagged_start - pd.Timedelta(days=5)
                plot_end = flagged_end + pd.Timedelta(days=10)
                plot_df = df[(df['Date/Time'] >= plot_start) & (df['Date/Time'] <= plot_end)].copy()
            else:
                plot_df = df.copy()
                plot_start = df['Date/Time'].min()
                plot_end = df['Date/Time'].max()

            # Convert columns to numeric
            for col in ['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6', 'RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']:
                if col in plot_df.columns:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

            # Reset index for Altair compatibility
            plot_df = plot_df.reset_index(drop=True)

            # Plot 1: Sensor Values
            st.markdown("#### Sensor Values Over Time")
            chart_data = plot_df[['Date/Time', 'RTD','Setpoint', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']].dropna()
            if not chart_data.empty:
                chart_data_melted = chart_data.melt('Date/Time', var_name='Sensor', value_name='Value')
                chart = alt.Chart(chart_data_melted).mark_line().encode(
                    x=alt.X('Date/Time:T', title='Date'),
                    y=alt.Y('Value:Q', title='Values'),
                    color='Sensor:N',
                    tooltip=['Date/Time:T', 'Sensor:N', 'Value:Q']
                ).properties(
                    width=800,
                    height=400
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No data available for this period.")

            # Plot 2: Sensor Trends
            st.markdown("#### Sensor Trends Over Time")
            chart_data = plot_df[['Date/Time', 'RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']].dropna()
            if not chart_data.empty:
                chart_data_melted = chart_data.melt('Date/Time', var_name='Trend', value_name='Value')
                chart = alt.Chart(chart_data_melted).mark_line().encode(
                    x=alt.X('Date/Time:T', title='Date'),
                    y=alt.Y('Value:Q', title='Trend Values'),
                    color='Trend:N',
                    tooltip=['Date/Time:T', 'Trend:N', 'Value:Q']
                ).properties(
                    width=800,
                    height=400
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No data available for this period.")

            # Plot 3: Random Forest Feature Importance
            st.markdown("#### Random Forest Feature Importance")
            importances = model.feature_importances_
            feature_names = features
            indices = np.argsort(importances)
            sorted_features = [feature_names[i] for i in indices]
            sorted_importances = importances[indices]
            importance_df = pd.DataFrame({
                'Feature': sorted_features,
                'Importance': sorted_importances
            })
            chart = alt.Chart(importance_df).mark_bar().encode(
                y=alt.Y('Feature:N', sort=None, title='Feature'),
                x=alt.X('Importance:Q', title='Importance'),
                tooltip=['Feature:N', 'Importance:Q']
            ).properties(
                width=800,
                height=400
            )
            st.altair_chart(chart, use_container_width=True)

            # Plot 4: Daily Actual Values
            st.markdown("#### Daily Actual Sensor Values")
            df.set_index('Date/Time', inplace=True)
            daily_df_actual = df[['RTD','Setpoint', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']].resample('D').mean().reset_index()
            if not daily_df_actual.empty:
                chart_data_melted = daily_df_actual.melt('Date/Time', var_name='Sensor', value_name='Value')
                chart = alt.Chart(chart_data_melted).mark_line().encode(
                    x=alt.X('Date/Time:T', title='Date'),
                    y=alt.Y('Value:Q', title='Values'),
                    color='Sensor:N',
                    tooltip=['Date/Time:T', 'Sensor:N', 'Value:Q']
                ).properties(
                    width=800,
                    height=400
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No daily data available.")

            # Plot 5: Daily Trend Values
            st.markdown("#### Daily Sensor Trend Values")
            daily_df_trend = df[['RTD_trend', 'TC1_trend', 'TC10_trend', 'TC3_trend', 'TC4_trend', 'TC6_trend']].resample('D').mean().reset_index()
            if not daily_df_trend.empty:
                chart_data_melted = daily_df_trend.melt('Date/Time', var_name='Trend', value_name='Value')
                chart = alt.Chart(chart_data_melted).mark_line().encode(
                    x=alt.X('Date/Time:T', title='Date'),
                    y=alt.Y('Value:Q', title='Trend Values'),
                    color='Trend:N',
                    tooltip=['Date/Time:T', 'Trend:N', 'Value:Q']
                ).properties(
                    width=800,
                    height=400
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No daily data available.")

            # Plot 6: Flagged Daily Actual Values
            st.markdown("#### Flagged Daily Actual Sensor Values")
            if 'flagged' in locals() and not flagged.empty:
                flagged.set_index('Date/Time', inplace=True)
                daily_flagged = flagged[['RTD', 'TC1', 'TC10', 'TC3', 'TC4', 'TC6']].resample('D').mean().reset_index()
                if not daily_flagged.empty:
                    chart_data_melted = daily_flagged.melt('Date/Time', var_name='Sensor', value_name='Value')
                    chart = alt.Chart(chart_data_melted).mark_line().encode(
                        x=alt.X('Date/Time:T', title='Date'),
                        y=alt.Y('Value:Q', title='Values'),
                        color='Sensor:N',
                        tooltip=['Date/Time:T', 'Sensor:N', 'Value:Q']
                    ).properties(
                        width=800,
                        height=400
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.write("No flagged daily data available.")
            else:
                st.write("✅ No Issue Detected — Device Operating Normally")
