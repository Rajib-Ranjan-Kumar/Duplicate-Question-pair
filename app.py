import streamlit as st
import helper
import pickle
import numpy as np

import os

BASE_DIR = os.path.dirname(__file__)



with open(os.path.join(BASE_DIR, "rf_model.pkl"), "rb") as f:
    model = pickle.load(f)


st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("Please enter both questions")
    else:
        # 🔥 extract features (DataFrame)
        df = helper.extract_features(q1, q2)
        print(df.shape)
        # ensure numeric (important for XGBoost)
        df = df.astype(float)

        # predict
        result = model.predict(df.values)[0]

        if result == 1:
            st.success(f'Duplicate Question')
        else:
            st.error(f'Not Duplicate Question')



