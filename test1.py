

import folium

import streamlit as st
import pandas as pd
import numpy as np
import datetime

from streamlit_folium import st_folium, folium_static
import warnings
warnings.filterwarnings('ignore')
st.title('Decision Support System: Road Safety in London')

DATE_COLUMN = 'date'

DATA_URL = (r"C:\Users\Admin\Documents\GitHub\test\2021_data_accident.csv")



@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
data_load_state = st.text('Loading data...')
data = load_data()
data = data.dropna(how='any',axis=0) 
data_load_state.text("Done!")
dataset = data
City_of_London = dataset[dataset['local_authority_highway'] == 'E09000001']
Westminster = dataset[dataset['local_authority_highway'] == 'E09000033']

data = pd.concat([City_of_London, Westminster], axis = 0)
vis = data.loc[:,['date','time','longitude','latitude']]
vis['time']= pd.to_datetime(vis['time'],format= '%H:%M' )

data2 = vis




from geopy.distance import great_circle

def greatcircle(x,y):
    lat1, long1 = x[0], x[1]
    lat2, long2 = y[0], y[1]
    dist = great_circle((lat1,long1),(lat2,long2)).meters
    return dist

from sklearn.cluster import DBSCAN as dbscan



with st.sidebar:
    
    date1 = st.date_input("From",key = "start",min_value  = datetime.date(2012, 1, 1), max_value = datetime.date.today(), value =  datetime.date(2012, 1, 1) )
    date2 = st.date_input("To",key = "finish",min_value  = date1, max_value = datetime.date.today())
    if st.checkbox('Add time element'):
        hour_to_filter = st.slider('From this hour', 0, 23, 0, key = 2)
        hour_end = st.slider('To this hour', 0, 24, 24,key = 1)
        st.subheader('Map of all accidents from %s:00 to %s:00' % (hour_to_filter, hour_end))

    else:
        hour_to_filter, hour_end = [0,24]
# Some number in the range 0-23

filtered_data = data2[(data2['time'].dt.hour >= hour_to_filter) & (data2['time'].dt.hour < hour_end)
&(data2['date'].dt.date >=date1) & (data2['date'].dt.date <= date2)]
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
    st.map(filtered_data)

# Using object notation
st.write(len(filtered_data.index))

dataset = filtered_data
max_cluster_distance = st.slider('Max distance', 10, 200, 150,key = 'ep',step = 5)
min_samples_in_cluster = st.slider('Min number', 2, 20, 12,key = 'minx')

location_data = dataset[['latitude','longitude']]

clusters = dbscan(eps = max_cluster_distance, min_samples = min_samples_in_cluster, metric=greatcircle).fit(location_data)

labels = clusters.labels_
unique_labels = np.unique(clusters.labels_)

dataset['Cluster'] = labels
# Using "with" notation
location = dataset['latitude'].mean(), dataset['longitude'].mean()

map_plot = folium.Map(location=location,zoom_start=13)
clust_colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
folium.TileLayer('cartodbpositron').add_to(map_plot)
for i in range(0,len(dataset)):
    colouridx = dataset['Cluster'].iloc[i]
    if colouridx == -1:
        pass
    else:
        col = clust_colours[colouridx%len(clust_colours)]
        folium.CircleMarker([dataset['latitude'].iloc[i],dataset['longitude'].iloc[i]], radius = 3, color = col, fill = col).add_to(map_plot)
        
st.divider()  # 👈 Draws a horizontal rule
st.subheader('Cluster accidents hotspot by location')
folium_static(map_plot)


import matplotlib.pyplot as plt
import seaborn as sns


road_cond = data['road_surface_conditions'].value_counts()

road_cond_arr = data['road_surface_conditions'].unique()
road_num_acc_arr = road_cond.values

weather_cond = data['weather_conditions'].value_counts() 

weather_cond_arr = data['weather_conditions'].unique()
weather_num_acc_arr = weather_cond.values
st.subheader('Accident Rate by each Factors')

plt.figure(figsize = (20,9),facecolor='grey')
plt.subplot(1, 2, 1)

plt.pie(road_num_acc_arr, labels = road_cond_arr, colors = sns.color_palette(),startangle = 30,textprops={'size': 'large'},explode=(0.02,0.02,0.02,0.02,0.3),autopct="%1.1f%%")
plt.legend()
plt.title("Accident Rate by Road Conditions",weight="bold")


plt.subplot(1, 2, 2)

plt.pie(weather_num_acc_arr, labels = weather_cond_arr,colors = sns.color_palette(),startangle = 30,textprops={'size': 'large'},explode=(0.01,0.01,0.01,0.01,0.01,0.20,0.3,0.50,0.7),autopct="%1.1f%%")
plt.legend(loc ="lower left")
plt.title("Accident Rate by Weather Conditions",weight="bold")

fig = plt.gcf()

st.pyplot(fig)

light_cond = data['light_conditions'].value_counts()

light_cond_arr = data['light_conditions'].unique()
light_num_acc_arr = light_cond.values

road_type = data['road_type'].value_counts() 

road_type_arr = data['road_type'].unique()
road_type_num_acc_arr = road_type.values

plt.figure(figsize = (20,9),facecolor='grey')
plt.subplot(1, 2, 1)

plt.pie(light_num_acc_arr, labels = light_cond_arr, colors = sns.color_palette(),startangle = 30,textprops={'size': 'large'},autopct="%1.1f%%")
plt.legend()
st.divider()  # 👈 Draws a horizontal rule

plt.title("Accident Rate by Light Conditions",weight="bold")


plt.subplot(1, 2, 2)

plt.pie(road_type_num_acc_arr, labels =road_type_arr,colors = sns.color_palette(),startangle = 30,textprops={'size': 'large'},autopct="%1.1f%%")
plt.legend(loc ="lower left")
plt.title("Accident Rate by Road Type",weight="bold")

fig = plt.gcf()

st.pyplot(fig)
st.subheader('Accidents by time')
fig = plt.figure(figsize=(10, 8))
sns.countplot(data,x="day_of_week")

st.pyplot(fig)
st.divider()  # 👈 Draws a horizontal rule


acc_by_time = dataset.time
acc_by_time['hour']  = pd.to_datetime(acc_by_time.values,format= '%H:%M' ).hour
fig = plt.figure(figsize=(10, 8))

sns.countplot(acc_by_time,x="hour")
st.pyplot(fig)
st.divider() 