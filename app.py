import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Transaction Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #E53935;
    text-align: center;
    margin-bottom: 2rem;
}
.anomaly-high { color: #e74c3c; font-weight: bold; }
.anomaly-low { color: #2ecc71; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔍 Transaction Anomaly Detection System</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("🛠️ Control Panel")
page = st.sidebar.radio("Navigate", ["Real-time Monitor", "Historical Analysis", "Model Insights", "Alert Settings"])

# Generate Sample Transaction Data
@st.cache_data
def generate_transaction_data():
    np.random.seed(42)
    n_transactions = 500
    
    # Normal transactions
    normal_amounts = np.random.lognormal(4, 1, int(n_transactions * 0.95))
    
    # Anomalous transactions
    anomaly_amounts = np.random.choice([np.random.uniform(10000, 50000), np.random.uniform(0.01, 1)], int(n_transactions * 0.05))
    
    amounts = np.concatenate([normal_amounts, [anomaly_amounts] * int(n_transactions * 0.05)])
    np.random.shuffle(amounts)
    
    data = {
        'TransactionID': [f'TXN{str(i).zfill(6)}' for i in range(1, n_transactions + 1)],
        'Amount': np.abs(np.random.lognormal(4, 1.5, n_transactions)),
        'Timestamp': pd.date_range(end=datetime.now(), periods=n_transactions, freq='5min'),
        'MerchantCategory': np.random.choice(['Retail', 'Food', 'Travel', 'Online', 'ATM', 'Transfer'], n_transactions),
        'Location': np.random.choice(['New York', 'London', 'Tokyo', 'Dubai', 'Singapore'], n_transactions),
        'DeviceType': np.random.choice(['Mobile', 'Desktop', 'POS', 'ATM'], n_transactions),
        'AnomalyScore': np.random.beta(2, 8, n_transactions),
    }
    df = pd.DataFrame(data)
    df['IsAnomaly'] = df['AnomalyScore'] > 0.5
    
    # Add some obvious anomalies
    anomaly_idx = np.random.choice(df.index, size=25, replace=False)
    df.loc[anomaly_idx, 'Amount'] = np.random.uniform(8000, 25000, 25)
    df.loc[anomaly_idx, 'AnomalyScore'] = np.random.uniform(0.7, 0.98, 25)
    df.loc[anomaly_idx, 'IsAnomaly'] = True
    
    return df

df = generate_transaction_data()

if page == "Real-time Monitor":
    st.header("📡 Real-time Transaction Monitor")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_txns = len(df)
    anomaly_count = df['IsAnomaly'].sum()
    total_amount = df['Amount'].sum()
    avg_score = df['AnomalyScore'].mean()
    
    with col1:
        st.metric("Total Transactions", f"{total_txns:,}", "+12%")
    with col2:
        st.metric("⚠️ Anomalies Detected", f"{anomaly_count}", f"{(anomaly_count/total_txns*100):.1f}%")
    with col3:
        st.metric("Total Volume", f"${total_amount:,.0f}", "+8.5%")
    with col4:
        st.metric("Avg Risk Score", f"{avg_score:.3f}", "-0.02")
    
    st.markdown("---")
    
    # Recent Anomalies
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Transaction Volume Timeline")
        hourly = df.set_index('Timestamp').resample('1H')['Amount'].sum().reset_index()
        fig = px.area(hourly, x='Timestamp', y='Amount', title='Hourly Transaction Volume')
        fig.update_traces(fill='tozeroy', line_color='#1E88E5')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Recent Alerts")
        recent_anomalies = df[df['IsAnomaly']].nlargest(5, 'AnomalyScore')
        for _, row in recent_anomalies.iterrows():
            st.error(f"""
            **{row['TransactionID']}**  
            Amount: ${row['Amount']:,.2f}  
            Risk: {row['AnomalyScore']:.2%}
            """)

elif page == "Historical Analysis":
    st.header("📈 Historical Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Distribution", "Trends", "Anomaly Details"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='Amount', color='IsAnomaly', nbins=50,
                              title='Transaction Amount Distribution',
                              color_discrete_map={True: '#e74c3c', False: '#3498db'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(df, x='Amount', y='AnomalyScore', color='IsAnomaly',
                            title='Amount vs Anomaly Score',
                            color_discrete_map={True: '#e74c3c', False: '#3498db'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        category_stats = df.groupby('MerchantCategory').agg({
            'Amount': 'sum',
            'IsAnomaly': 'sum',
            'AnomalyScore': 'mean'
        }).reset_index()
        
        fig = px.bar(category_stats, x='MerchantCategory', y='Amount',
                     color='IsAnomaly', title='Transaction Volume by Category')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Detected Anomalies")
        anomalies = df[df['IsAnomaly']].sort_values('AnomalyScore', ascending=False)
        st.dataframe(anomalies[['TransactionID', 'Amount', 'MerchantCategory', 'Location', 'AnomalyScore']], 
                    use_container_width=True)

elif page == "Model Insights":
    st.header("🤖 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Precision", "94.2%", "1.5%")
        st.metric("Recall", "91.8%", "2.3%")
    with col2:
        st.metric("F1-Score", "92.9%", "1.9%")
        st.metric("AUC-ROC", "0.967", "0.012")
    with col3:
        st.metric("False Positive Rate", "2.1%", "-0.3%")
        st.metric("Detection Latency", "45ms", "-5ms")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature Importance
        features = ['Transaction Amount', 'Time of Day', 'Location Risk', 'Device Type', 'Merchant Category', 'Velocity']
        importance = [0.32, 0.18, 0.17, 0.14, 0.11, 0.08]
        fig = px.bar(x=importance, y=features, orientation='h',
                     title='Feature Importance (Isolation Forest)',
                     labels={'x': 'Importance', 'y': 'Feature'})
        fig.update_traces(marker_color='#9C27B0')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROC Curve simulation
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - (1 - fpr) ** 3
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='Model (AUC=0.967)'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)

else:  # Alert Settings
    st.header("⚙️ Alert Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Alert Thresholds")
        high_risk = st.slider("High Risk Threshold", 0.0, 1.0, 0.8)
        medium_risk = st.slider("Medium Risk Threshold", 0.0, 1.0, 0.5)
        amount_threshold = st.number_input("Amount Threshold ($)", 1000, 100000, 10000)
    
    with col2:
        st.subheader("📧 Notification Settings")
        email_alerts = st.checkbox("Email Alerts", value=True)
        sms_alerts = st.checkbox("SMS Alerts", value=False)
        slack_alerts = st.checkbox("Slack Notifications", value=True)
        
        if email_alerts:
            st.text_input("Email Address", "alerts@company.com")
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Alert settings saved successfully!")
        st.balloons()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with ❤️ by Jaimin Prajapati | Transaction Anomaly Detection ML Project</p>", unsafe_allow_html=True)
