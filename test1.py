


import streamlit as st
import pandas as pd
import numpy as np
import datetime
st.title('Accidents in UK')

DATE_COLUMN = 'date/time'
DATA_URL = (r"C:\Users\Admin\Documents\GitHub\test\accident_data.csv")



@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(1000)
data_load_state.text("Done!")

vis = data.loc[:,['date/time','time','longitude','latitude']]
vis['time']= pd.to_datetime(vis['time'],format= '%H:%M' )

#vis.set_index(['date/time'], inplace = True)
data2 = vis
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data2)




# Using "with" notation
with st.sidebar:
    
    date1 = st.date_input("from",key = "start",min_value  = datetime.date(2012, 1, 1), max_value = datetime.date.today())
    date2 = st.date_input("to",key = "finish",min_value  = date1, max_value = datetime.date.today())
    hour_to_filter = st.slider('hour from', 0, 23, 17, key = 2)
    hour_end = st.slider('hour to', 0, 23, 17,key = 1)
# Some number in the range 0-23

filtered_data = data2[(data2['time'].dt.hour >= hour_to_filter) & (data2['time'].dt.hour <= hour_end)
&(data['date/time'].dt.date >=date1) & (data['date/time'].dt.date <= date2)]

st.subheader('Map of all accidents from %s:00 to %s:00' % (hour_to_filter, hour_end))
st.map(filtered_data)
# Using object notation
st.write(len(filtered_data.index))
