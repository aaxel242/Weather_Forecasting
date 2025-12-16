import streamlit as st
import prophet as prophet

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from utils.evaluate import evaluate_classification, evaluate_regression

def prediction_model_train(df):
    st.markdown("---")
    st.title("Prediction with Random Forest")
    
    x = df[["ram", "battery_power", "px_height", "px_width"]]
    y = df["price_range"]

    # split = int(len(df)*0.8)
    # x_train, x_test = x.iloc[:split], x.iloc[split:]
    # y_train, y_test = y.iloc[:split], y.iloc[split:]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Prediction with RandomForestRegressor()
    st.subheader("Regression")
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    st.write(f"MAE: {mean_absolute_error(y_test, pred)}")

    results_regression = evaluate_regression(y_test, pred)

    st.write(results_regression)

    # Prediction with RandomForestClassifier()
    st.subheader("Classification")
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)

    pred = model.predict(x_test)
    st.write(f"MAE (Mean Absolute Error): {mean_absolute_error(y_test, pred)}")

    results_clasifiaction = evaluate_classification(y_test, pred)

    st.write(results_clasifiaction)
    
# este no se utiliza de momento porque se necesita datos con fechas
def prediction_model(df):

    df_prophet = df.rename(columns={"ram": "ds", "price_range": "y"})
    m = prophet.Prophet()
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    m.plot(forecast)