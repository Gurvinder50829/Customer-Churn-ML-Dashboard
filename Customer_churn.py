# ===========================================
# IMPORT LIBRARIES
# ===========================================

import pandas as pd
import joblib
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===========================================
# LOAD DATASET
# ===========================================

df = pd.read_csv(
    r"C:\Users\Gurvinder Singh\Desktop\Customer_churn\WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# ===========================================
# DATA PREPROCESSING
# ===========================================

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Drop CustomerID
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Encode categorical columns
encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# ===========================================
# FEATURES & TARGET
# ===========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ===========================================
# TRAIN TEST SPLIT
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ===========================================
# STANDARD SCALER
# ===========================================

Scaler = StandardScaler()

X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)

# Save Scaler
joblib.dump(Scaler, "scaler.pkl")

print("✅ Scaler Saved Successfully")

# ===========================================
# RANDOM FOREST MODEL
# ===========================================

Model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
Model.fit(X_train, y_train)

print("✅ Model Training Completed")

# ===========================================
# PREDICTION
# ===========================================

Prediction = Model.predict(X_test)

accuracy = accuracy_score(y_test, Prediction)

print(f"Accuracy : {accuracy:.4f}")

# ===========================================
# SAVE MODEL
# ===========================================

joblib.dump(Model, "churn_model.pkl")

print("✅ Model Saved Successfully")

# ===========================================
# LOAD MODEL
# ===========================================

Model1 = joblib.load("churn_model.pkl")
Scaler1 = joblib.load("scaler.pkl")

print("✅ Model Loaded Successfully")
print("✅ Scaler Loaded Successfully")

# ===========================================
# TEST LOADED MODEL
# ===========================================

Prediction2 = Model1.predict(X_test)

print(
    "Loaded Model Accuracy :",
    accuracy_score(y_test, Prediction2)
)

print("===================================")
print("Customer Churn Model Ready")
print("Files Created:")
print("1. churn_model.pkl")
print("2. scaler.pkl")
print("===================================")
# ===========================================
# CUSTOMER CHURN PREDICTION SYSTEM
# STREAMLIT APP - PART 1
# ===========================================

# ==========================================
# CUSTOMER CHURN PREDICTION SYSTEM
# HOME PAGE
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==========================================
st.markdown("""

<style>

/* Full Screen Dashboard Width */

.block-container{

    padding-top:1rem;

    padding-left:2rem;

    padding-right:2rem;

    max-width:100%;

}



/* Plotly Full Width */

div[data-testid="stPlotlyChart"]{

    width:100% !important;

}



div[data-testid="stPlotlyChart"] > div{

    width:100% !important;

}



/* Remove chart extra margins */

.js-plotly-plot{

    width:100% !important;

}


</style>

""", unsafe_allow_html=True)
colors = [
    "#00E5FF",
    "#00C853",
    "#FFD600",
    "#FF6D00",
    "#D500F9",
    "#FF1744",
    "#00BFA5",
    "#2979FF",
    "#FFEA00"
]

# ==========================================
# LOAD DATASET
# ==========================================

Data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges

Data["TotalCharges"] = pd.to_numeric(
    Data["TotalCharges"],
    errors="coerce"
)

Data["TotalCharges"].fillna(
    Data["TotalCharges"].median(),
    inplace=True
)
# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📊 Customer Churn")

Menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Dataset",
        "📊 Data Visualization",
        "🤖 Machine Learning",
        "🔮 Churn Prediction",
    ]
)

# ==========================================
# HOME PAGE
# ==========================================

# ===============================
# Row 1
# ===============================
# ==========================================
# 🏠 HOME PAGE
# ==========================================

if Menu == "🏠 Home":

    # Main Title
    st.title("📊 Customer Churn Prediction Dashboard")

    st.caption(
        "AI-powered analytics dashboard to analyze customer behavior "
        "and predict customer churn using Machine Learning."
    )


    st.markdown("---")


    # ==========================================
    # PROJECT OVERVIEW
    # ==========================================

    st.header("📌 Project Overview")

    st.write(
        """
        This dashboard provides complete insights into the IBM Telco Customer 
        Churn dataset. It includes customer analysis, visualization, 
        machine learning prediction and model evaluation.
        """
    )


    st.markdown("---")


    # ==========================================
    # KPI SECTION
    # ==========================================

    st.header("📈 Dataset Key Performance Indicators")


    Total_Customers = len(Data)

    Total_Features = Data.shape[1]

    Churn_Customers = Data[
        Data["Churn"]=="Yes"
    ].shape[0]


    Active_Customers = Data[
        Data["Churn"]=="No"
    ].shape[0]


    Avg_MonthlyCharges = round(
        Data["MonthlyCharges"].mean(),
        2
    )


    Avg_Tenure = round(
        Data["tenure"].mean(),
        2
    )



    col1,col2,col3,col4,col5,col6 = st.columns(6)


    with col1:
        with st.container(border=True):
            st.metric(
                "👥 Total Customers",
                Total_Customers
            )


    with col2:
        with st.container(border=True):
            st.metric(
                "📑 Total Features",
                Total_Features
            )


    with col3:
        with st.container(border=True):
            st.metric(
                "❌ Churn Customers",
                Churn_Customers
            )


    with col4:
        with st.container(border=True):
            st.metric(
                "✅ Active Customers",
                Active_Customers
            )


    with col5:
        with st.container(border=True):
            st.metric(
                "💰 Avg Monthly Charges",
                Avg_MonthlyCharges
            )


    with col6:
        with st.container(border=True):
            st.metric(
                "📅 Avg Tenure",
                Avg_Tenure
            )


    st.markdown("---")



    # ==========================================
    # CUSTOMER ANALYTICS
    # ==========================================

    st.header("📊 Customer Analytics Visualization")

    st.subheader(
        "Interactive Churn Analysis Dashboard"
    )



    # ==============================
    # PLOT ROW 1
    # ==============================

    c1,c2,c3 = st.columns(3)



    with c1:

        with st.container(border=True):

            fig = px.histogram(
                Data,
                x="Churn",
                color="Churn",
                title="Customer Churn Distribution",
                text_auto=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )




    with c2:

        with st.container(border=True):

            fig = px.histogram(
                Data,
                x="gender",
                color="gender",
                title="Gender Distribution",
                text_auto=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )



    with c3:

        with st.container(border=True):

            fig = px.histogram(
                Data,
                x="Contract",
                color="Contract",
                title="Contract Type Analysis",
                text_auto=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )



    # ==============================
    # PLOT ROW 2
    # ==============================


    c4,c5,c6 = st.columns(3)



    with c4:

        with st.container(border=True):

            fig = px.pie(
                Data,
                names="InternetService",
                title="Internet Service Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )



    with c5:

        with st.container(border=True):

            payment = Data[
                "PaymentMethod"
            ].value_counts().reset_index()


            payment.columns=[
                "PaymentMethod",
                "Count"
            ]


            fig = px.bar(
                payment,
                x="PaymentMethod",
                y="Count",
                title="Payment Method"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )



    with c6:

        with st.container(border=True):

            fig = px.histogram(
                Data,
                x="MonthlyCharges",
                nbins=30,
                title="Monthly Charges Distribution"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )



    # ==============================
    # PLOT ROW 3
    # ==============================


    c7,c8,c9 = st.columns(3)



    with c7:

        with st.container(border=True):

            fig = px.histogram(
                Data,
                x="TotalCharges",
                nbins=30,
                title="Total Charges Distribution"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )



    with c8:

        with st.container(border=True):

            fig = px.histogram(
                Data,
                x="tenure",
                nbins=30,
                title="Customer Tenure Analysis"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )



    with c9:

        with st.container(border=True):

            fig = px.imshow(
                Data.corr(numeric_only=True),
                text_auto=True,
                title="Feature Correlation Heatmap"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )



    st.success(
        "✅ Customer Churn Dashboard Loaded Successfully"
    )
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
if Menu == "📂 Dataset":

    # ==========================================
    # DATASET DASHBOARD
    # ==========================================

    st.title("📂 Customer Dataset Dashboard")

    st.caption(
        "Comprehensive overview of the IBM Telco Customer Churn Dataset"
    )

    st.markdown("---")


    # ==========================================
    # LOAD DATA
    # ==========================================

    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.dropna(inplace=True)



    # ==========================================
    # KPI SECTION
    # ==========================================

    st.header("📊 Dataset Overview")


    col1,col2,col3,col4,col5,col6 = st.columns(6)


    with col1:
        st.metric(
            "Total Customers",
            f"{df.shape[0]:,}"
        )


    with col2:
        st.metric(
            "Total Features",
            df.shape[1]
        )


    with col3:
        st.metric(
            "Churn Customers",
            df[df["Churn"]=="Yes"].shape[0]
        )


    with col4:
        st.metric(
            "Retention Customers",
            df[df["Churn"]=="No"].shape[0]
        )


    with col5:
        st.metric(
            "Average Tenure",
            f"{df['tenure'].mean():.1f} Months"
        )


    with col6:
        st.metric(
            "Average Charges",
            f"${df['MonthlyCharges'].mean():.2f}"
        )



    st.markdown("---")


    # ==========================================
    # DATA PREVIEW
    # ==========================================

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    st.markdown("---")



    # ==========================================
    # PLOTLY SUBPLOTS
    # ==========================================


    st.header("📈 Dataset Analysis Visualization")


    fig = make_subplots(

        rows=3,
        cols=3,

        subplot_titles=(

            "Churn Distribution",
            "Gender Distribution",
            "Contract Type",

            "Internet Service",
            "Payment Method",
            "Customer Tenure",

            "Monthly Charges",
            "Total Charges",
            "Senior Citizen"

        ),

        specs=[

            [
                {"type":"pie"},
                {"type":"bar"},
                {"type":"bar"}
            ],

            [
                {"type":"bar"},
                {"type":"bar"},
                {"type":"histogram"}
            ],

            [
                {"type":"histogram"},
                {"type":"histogram"},
                {"type":"pie"}
            ]

        ]

    )



    # 1 Churn Pie

    churn = df["Churn"].value_counts()


    fig.add_trace(

        go.Pie(
            labels=churn.index,
            values=churn.values,
            hole=0.4
        ),

        row=1,
        col=1
    )



    # 2 Gender

    gender = df["gender"].value_counts()


    fig.add_trace(

        go.Bar(
            x=gender.index,
            y=gender.values,
            text=gender.values
        ),

        row=1,
        col=2
    )



    # 3 Contract

    contract = df["Contract"].value_counts()


    fig.add_trace(

        go.Bar(
            x=contract.index,
            y=contract.values
        ),

        row=1,
        col=3
    )



    # 4 Internet Service

    internet = df["InternetService"].value_counts()


    fig.add_trace(

        go.Bar(
            x=internet.index,
            y=internet.values
        ),

        row=2,
        col=1
    )



    # 5 Payment Method

    payment = df["PaymentMethod"].value_counts()


    fig.add_trace(

        go.Bar(
            x=payment.index,
            y=payment.values
        ),

        row=2,
        col=2
    )



    # 6 Tenure Distribution

    fig.add_trace(

        go.Histogram(
            x=df["tenure"],
            nbinsx=30
        ),

        row=2,
        col=3
    )



    # 7 Monthly Charges

    fig.add_trace(

        go.Histogram(
            x=df["MonthlyCharges"],
            nbinsx=40
        ),

        row=3,
        col=1
    )



    # 8 Total Charges

    fig.add_trace(

        go.Histogram(
            x=df["TotalCharges"],
            nbinsx=40
        ),

        row=3,
        col=2
    )



    # 9 Senior Citizen

    senior = df["SeniorCitizen"].value_counts()


    fig.add_trace(

        go.Pie(

            labels=["No","Yes"],
            values=senior.values,
            hole=0.4

        ),

        row=3,
        col=3
    )



    # ==========================================
    # FIGURE DESIGN
    # ==========================================

    fig.update_layout(

        height=1000,

        title_text="Customer Churn Dataset Analytics",

        template="plotly_dark",

        showlegend=True,

        paper_bgcolor="#111827",

        plot_bgcolor="#111827"

    )


    fig.update_xaxes(
        showgrid=False
    )


    fig.update_yaxes(
        showgrid=False
    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )

if Menu == "📊 Data Visualization":

    # ==========================================
    # DATA VISUALIZATION PAGE
    # ==========================================

    st.title("📊 Customer Churn Data Visualization")

    st.caption(
        "Interactive visual analysis of customer behavior and churn patterns."
    )

    st.markdown("---")


    # Load Dataset

    df = pd.read_csv(
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )


    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.dropna(inplace=True)



    # ==========================================
    # KPI SECTION
    # ==========================================

    st.header("📌 Visualization Summary")


    col1,col2,col3,col4 = st.columns(4)


    with col1:
        st.metric(
            "Customers",
            df.shape[0]
        )


    with col2:
        churn_rate = (
            df["Churn"].value_counts(normalize=True)["Yes"]*100
        )

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )


    with col3:
        st.metric(
            "Avg Monthly Charges",
            f"${df['MonthlyCharges'].mean():.2f}"
        )


    with col4:
        st.metric(
            "Avg Tenure",
            f"{df['tenure'].mean():.1f} Months"
        )


    st.markdown("---")



    # ==========================================
    # ROW 1
    # ==========================================


    col1,col2 = st.columns(2)


    with col1:

        st.subheader("🔥 Churn Distribution")


        churn = df["Churn"].value_counts()


        fig = px.pie(

            values=churn.values,

            names=churn.index,

            hole=0.45,

            title="Customer Churn"

        )


        fig.update_layout(
            template="plotly_dark"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )



    with col2:

        st.subheader("👥 Customer Gender")


        gender = (
            df["gender"]
            .value_counts()
            .reset_index()
        )


        gender.columns=[
            "Gender",
            "Count"
        ]


        fig = px.bar(

            gender,

            x="Gender",

            y="Count",

            text="Count",

            title="Gender Distribution"

        )


        fig.update_layout(
            template="plotly_dark"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )



    # ==========================================
    # ROW 2
    # ==========================================


    col1,col2 = st.columns(2)


    with col1:


        st.subheader("📄 Contract vs Churn")


        contract = (

            df.groupby(
                ["Contract","Churn"]
            )
            .size()
            .reset_index(
                name="Customers"
            )

        )


        fig = px.bar(

            contract,

            x="Contract",

            y="Customers",

            color="Churn",

            barmode="group"

        )


        fig.update_layout(
            template="plotly_dark"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )



    with col2:


        st.subheader("🌐 Internet Service")


        internet = df["InternetService"].value_counts()


        fig = px.bar(

            x=internet.index,

            y=internet.values,

            labels={
                "x":"Internet Service",
                "y":"Customers"
            }

        )


        fig.update_layout(
            template="plotly_dark"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )



    # ==========================================
    # ROW 3
    # ==========================================


    col1,col2 = st.columns(2)



    with col1:


        st.subheader("💰 Monthly Charges Distribution")


        fig = px.histogram(

            df,

            x="MonthlyCharges",

            nbins=40,

            title="Monthly Charges"

        )


        fig.update_layout(
            template="plotly_dark"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )



    with col2:


        st.subheader("⏳ Tenure Distribution")


        fig = px.histogram(

            df,

            x="tenure",

            nbins=35,

            title="Customer Tenure"

        )


        fig.update_layout(
            template="plotly_dark"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )



    # ==========================================
    # CORRELATION HEATMAP
    # ==========================================


    st.markdown("---")

    st.subheader("🔥 Feature Correlation")


    numeric_df = df.select_dtypes(
        include="number"
    )


    corr = numeric_df.corr()


    fig = px.imshow(

        corr,

        text_auto=True,

        title="Correlation Matrix",

        color_continuous_scale="RdBu"

    )


    fig.update_layout(
        template="plotly_dark",
        height=700
    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )
    # Prediction
Prediction = Model.predict(X_test)

# Probability for ROC
Probability = Model.predict_proba(X_test)[:,1]
# ==========================================
# MODEL SCORE PLOT
# ==========================================
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
accuracy = accuracy_score(y_test, Prediction)
precision = precision_score(y_test, Prediction)
recall = recall_score(y_test, Prediction)
f1 = f1_score(y_test, Prediction)
roc_auc = roc_auc_score(y_test, Probability)
import numpy as np
import joblib


# after train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model

Model.fit(
    X_train,
    y_train
)


# Save model

joblib.dump(
    Model,
    "churn_model.pkl"
)


joblib.dump(
    Scaler,
    "scaler.pkl"
)


# Save testing data

joblib.dump(
    X_test,
    "X_test.pkl"
)


joblib.dump(
    y_test,
    "y_test.pkl"
)


print("All Files Saved")

if Menu == "🤖 Machine Learning":

    st.title(
        "🤖Machine Learning Prediction Dashboard"
    )


    st.caption(
        "Machine Learning model evaluation and customer churn detection"
    )


    st.markdown("---")



    # ==========================================
    # LOAD MODEL FILES


    # ==========================================
    # MODEL PREDICTION
    # ==========================================


    Prediction = Model.predict(
        X_test
    )


    Probability = Model.predict_proba(
        X_test
    )[:,1]



    # ==========================================
    # PERFORMANCE METRICS
    # ==========================================


    accuracy = accuracy_score(
        y_test,
        Prediction
    )


    precision = precision_score(
        y_test,
        Prediction
    )


    recall = recall_score(
        y_test,
        Prediction
    )


    f1 = f1_score(
        y_test,
        Prediction
    )


    roc_auc = roc_auc_score(
        y_test,
        Probability
    )



    st.header(
        "📊 Model Performance"
    )


    col1,col2,col3,col4,col5 = st.columns(5)



    col1.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )


    col2.metric(
        "Precision",
        f"{precision*100:.2f}%"
    )


    col3.metric(
        "Recall",
        f"{recall*100:.2f}%"
    )


    col4.metric(
        "F1 Score",
        f"{f1*100:.2f}%"
    )


    col5.metric(
        "ROC-AUC",
        f"{roc_auc:.3f}"
    )



    # ==========================================
    # SCORE VISUALIZATION
    # ==========================================


    st.subheader(
        "📈 Model Score Analysis"
    )


    score_df = pd.DataFrame(
        {
            "Metric":
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],

            "Score":
            [
                accuracy,
                precision,
                recall,
                f1,
                roc_auc
            ]
        }
    )


    fig = px.bar(
        score_df,
        x="Metric",
        y="Score",
        text="Score",
        title="ML Model Performance"
    )


    fig.update_traces(
        texttemplate="%{text:.3f}"
    )


    fig.update_layout(
        template="plotly_dark",
        yaxis_range=[0,1]
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )



    # ==========================================
    # CONFUSION MATRIX
    # ==========================================


    st.subheader(
        "📌 Confusion Matrix"
    )


    cm = confusion_matrix(
        y_test,
        Prediction
    )


    fig = px.imshow(
        cm,
        text_auto=True,
        labels=
        {
            "x":"Predicted",
            "y":"Actual"
        },
        title="Confusion Matrix"
    )


    fig.update_layout(
        template="plotly_dark"
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )



    # ==========================================
    # CLASSIFICATION REPORT
    # ==========================================


    st.subheader(
        "📄 Classification Report"
    )


    report = classification_report(
        y_test,
        Prediction,
        output_dict=True
    )


    report_df = pd.DataFrame(
        report
    ).T



    st.dataframe(
        report_df,
        width="stretch"
    )



    # ==========================================
    # ROC CURVE
    # ==========================================


    st.subheader(
        "📈 ROC Curve"
    )


    fpr,tpr,_ = roc_curve(
        y_test,
        Probability
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="ROC Curve"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=[0,1],
            y=[0,1],
            mode="lines",
            name="Random"
        )
    )


    fig.update_layout(
        title=f"ROC Curve AUC={roc_auc:.3f}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark"
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )



    # ==========================================
    # LIVE CUSTOMER CHURN DETECTION
    # ==========================================
if  Menu == "🔮 Churn Prediction":



    # ==========================================
    # CUSTOMER CHURN PREDICTION PAGE
    # ==========================================


    st.title("🔮 Customer Churn Prediction")


    st.caption(
        "Predict whether a customer will leave the service using Machine Learning."
    )


    st.markdown("---")



    # ==========================================
    # LOAD MODEL
    # ==========================================


    Model = joblib.load(
        "churn_model.pkl"
    )


    Scaler = joblib.load(
        "scaler.pkl"
    )



    # ==========================================
    # CUSTOMER INPUT
    # ==========================================


    st.header("👤 Customer Information")



    col1,col2,col3 = st.columns(3)



    with col1:


        gender = st.selectbox(

            "Gender",

            [
                "Male",
                "Female"
            ]

        )


        senior = st.selectbox(

            "Senior Citizen",

            [
                0,
                1
            ]

        )


        partner = st.selectbox(

            "Partner",

            [
                "Yes",
                "No"
            ]

        )


        dependents = st.selectbox(

            "Dependents",

            [
                "Yes",
                "No"
            ]

        )




    with col2:


        tenure = st.slider(

            "Tenure (Months)",

            0,

            72,

            12

        )


        phone = st.selectbox(

            "Phone Service",

            [
                "Yes",
                "No"
            ]

        )


        internet = st.selectbox(

            "Internet Service",

            [
                "DSL",
                "Fiber optic",
                "No"
            ]

        )


        contract = st.selectbox(

            "Contract",

            [
                "Month-to-month",
                "One year",
                "Two year"
            ]

        )





    with col3:


        monthly = st.number_input(

            "Monthly Charges",

            min_value=0.0,

            value=70.0

        )


        total = st.number_input(

            "Total Charges",

            min_value=0.0,

            value=2000.0

        )


        payment = st.selectbox(

            "Payment Method",

            [

            "Electronic check",

            "Mailed check",

            "Bank transfer (automatic)",

            "Credit card (automatic)"

            ]

        )


        paperless = st.selectbox(

            "Paperless Billing",

            [
                "Yes",
                "No"
            ]

        )



    st.markdown("---")



    # ==========================================
    # CREATE INPUT DATAFRAME
    # ==========================================



    if st.button(
        "🚀 Predict Churn"
    ):


        input_data = pd.DataFrame(

            {

            "gender":[gender],

            "SeniorCitizen":[senior],

            "Partner":[partner],

            "Dependents":[dependents],

            "tenure":[tenure],

            "PhoneService":[phone],

            "InternetService":[internet],

            "Contract":[contract],

            "MonthlyCharges":[monthly],

            "TotalCharges":[total],

            "PaymentMethod":[payment],

            "PaperlessBilling":[paperless]

            }

        )



        # Encoding

        input_data = pd.get_dummies(
            input_data
        )



        # Match training columns

        model_columns = Scaler.feature_names_in_


        for col in model_columns:

            if col not in input_data.columns:

                input_data[col] = 0



        input_data = input_data[
            model_columns
        ]



        # Scaling

        input_scaled = Scaler.transform(

            input_data

        )



        # Prediction

        prediction = Model.predict(

            input_scaled

        )


        probability = Model.predict_proba(

            input_scaled

        )[0][1]




        # ==========================================
        # RESULT
        # ==========================================


        st.markdown("---")


        st.subheader(
            "Prediction Result"
        )



        col1,col2 = st.columns(2)



        with col1:


            if prediction[0] == 1:


                st.error(
                    "⚠️ Customer Likely To Churn"
                )


            else:


                st.success(
                    "✅ Customer Will Stay"
                )



        with col2:


            st.metric(

                "Churn Probability",

                f"{probability*100:.2f}%"

            )



        # Probability Bar


        st.progress(

            float(probability)

        )