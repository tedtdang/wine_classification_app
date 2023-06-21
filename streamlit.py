import pandas as pd

import streamlit as st
from Processing import Processor

data_path = 'winequality-red.csv'
data_instance = Processor(data_path=data_path, target_column='quality')
st.title("Predict wine quality!")
st.sidebar.header('User input features')
# Collect user input features into dataframe
upload_file = st.sidebar.file_uploader('Upload your file here')
if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
    def user_input_features():
        features = pd.DataFrame()
        data = dict()
        for col in data_instance.X.columns:
            if col in data_instance.num_cols:
                # Calculate the critical values
                q1 = float(data_instance.df[col].quantile(0.25))
                median = float(data_instance.df[col].quantile(0.5))
                q3 = float(data_instance.df[col].quantile(0.75))
                iqr = q3 - q1
                upper_whisker = q3 + 1.5 * iqr
                lower_whisker = q1 - 1.5 * iqr
                sidebar = st.sidebar.slider(col, min_value=lower_whisker, max_value=upper_whisker, value=median)
            else:
                sidebar = st.sidebar.selectbox(col, *[data_instance.df[col].unique()])
            data[col] = sidebar
        features = pd.concat([pd.DataFrame(data, columns=data.keys(), index=[0]), features], ignore_index=True)
        return features


    input_df = user_input_features()

# Display the user input features
st.subheader('User input features')

if upload_file is None:
    st.write('Awaiting csv file to be uploaded. Currently using example input parameters shown below.')
st.write(input_df)

# Prediction
processor, model = data_instance.process()
input_df_transformed = processor.transform(input_df)
# labels = data_instance.encoder.classes_
y_pred = model.predict(input_df_transformed)
# y_pred = data_instance.encoder.inverse_transform(model.predict(input_df_transformed))
prediction_proba = pd.DataFrame(model.predict_proba(input_df_transformed), columns=sorted(data_instance.y.unique()))

st.subheader('Prediction')
st.write(f'Predicted value: {y_pred[0]}')
st.subheader('Prediction probability')
st.write(prediction_proba)
