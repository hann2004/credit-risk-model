import streamlit as st


def calculate_roi(
    loans_processed=1000,
    avg_loan=10000,
    default_rate=0.05,
    model_accuracy=0.8,
    dev_cost=50000,
):
    """
    Calculate business impact in dollars
    """
    baseline_losses = loans_processed * avg_loan * default_rate
    model_losses = baseline_losses * (
        1 - model_accuracy
    )  # e.g. 80% accuracy = 20% losses remain
    savings = baseline_losses - model_losses
    roi = ((savings - dev_cost) / dev_cost) * 100

    st.metric("Projected Annual Savings", f"${savings:,.0f}")
    st.metric("ROI", f"{roi:.0f}%", help="Based on $50K development cost")
    st.caption(
        f"Assumes {loans_processed} loans/year, avg loan ${avg_loan}, default rate {default_rate*100:.1f}%."
    )


st.title("💰 Business ROI Calculator")
st.write("Estimate the financial impact of deploying the credit risk model.")

loans = st.number_input("Loans processed per year", 100, 100000, 10000, 1000)
avg_loan = st.number_input("Average loan amount ($)", 1000, 100000, 10000, 1000)
default_rate = st.slider("Default rate (before model)", 0.01, 0.20, 0.05, 0.01)
model_acc = st.slider("Model accuracy (recall)", 0.5, 1.0, 0.8, 0.01)
dev_cost = st.number_input("Development cost ($)", 10000, 200000, 50000, 1000)

calculate_roi(loans, avg_loan, default_rate, model_acc, dev_cost)
