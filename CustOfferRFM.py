# Import libraries
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import squarify
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Read dataset
transaction_df = pd.read_csv('transaction_data.csv')
#transaction_df.to_csv("transaction_df.csv")
transaction_df.drop(['offer_target_age'], 1, inplace=True)

cust_df = pd.read_csv("Sample_Data_Offer_Customer_Transaction\customer_data.csv")
#cust_df.to_csv("cust_df.csv")
cust_df.drop(['member_title','segment','member_age','cust_tier'], 1,
             inplace=True)
#cust_df.drop(['customer_title','customer_maritalStatus','CustomerAgeSegment','CustomerAge','CustomerTier'], 1,inplace=True)

offer_df = pd.read_csv("Sample_Data_Offer_Customer_Transaction\offer_data.csv",encoding= 'unicode_escape')
#offer_df.to_csv("offer_df.csv")
offer_df.drop(['targetted_marital_status','targetted_tier','targetted_gender','targetted_age'],1,inplace=True)

#joining the data
joinedData_df = cust_df.merge(transaction_df, on = 'customer_id',how ='inner')
#joinedData_df.to_csv("Join.csv" , index=False)

def Rename_OfferTier(df):
    for col in ['customer_tier']:
        df[col] = df[col].map({'B':'Bronze', 'S':'Saphire','P':'Platinum','G':'Gold'})
    return df

joinedData_df = Rename_OfferTier(joinedData_df)
offerTransaction = offer_df.merge(joinedData_df, on='offer_id',how='inner')
#offerTransaction.to_excel("DataFull.xlsx")
# Convert InvoiceDate from object to datetime format
offerTransaction['invoice_date'] = pd.to_datetime(offerTransaction['offer_transaction_date'])

# Drop NA values from online
offerTransaction.dropna()

# --Group data by customerID--
# Create snapshot date
snapshot_date = offerTransaction['invoice_date'].max() + timedelta(days=1)
print(snapshot_date)

# Grouping by CustomerID
data_process = offerTransaction.groupby(['customer_id'],as_index=False).agg({
        'invoice_date': lambda x: (snapshot_date - x.max()).days,
        'offer_id': 'count',
        'transaction_amount': 'sum'})
print(data_process.columns.tolist())
# Rename the columns
data_process.rename(columns={'invoice_date': 'Recency',
                         'offer_id': 'Frequency',
                         'transaction_amount': 'MonetaryValue'}, inplace=True)

# Print top 20 rows and shape of dataframe
print(data_process.columns.tolist())


recency = data_process['Recency'].to_numpy()
frequency = data_process['Frequency'].to_numpy()
monetaryValue = data_process['MonetaryValue'].to_numpy()

#RECENCY
# data-structure to store Sum-Of-Square-Errors
sse = {}
# Looping over multiple values of k from 1 to 30
for k in range(0, 10):
    kmeans = KMeans(n_clusters=k+1).fit(recency.reshape(-1,1))
    sse[k] = kmeans.inertia_
# Plotting the curve with 'k'-value vs SSE
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster for Recency")
plt.ylabel("SSE")
# Save the Plot in current directory
plt.show()

#FREQUENCY
# data-structure to store Sum-Of-Square-Errors
sse = {}
# Looping over multiple values of k from 1 to 30
for k in range(0, 10):
    kmeans = KMeans(n_clusters=k+1).fit(frequency.reshape(-1,1))
    sse[k] = kmeans.inertia_
# Plotting the curve with 'k'-value vs SSE
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster for Frequency")
plt.ylabel("SSE")
# Save the Plot in current directory
plt.show()

#MONETARY VALUE
# data-structure to store Sum-Of-Square-Errors
sse = {}
# Looping over multiple values of k from 1 to 30
for k in range(0, 10):
    kmeans = KMeans(n_clusters=k+1).fit(monetaryValue.reshape(-1,1))
    sse[k] = kmeans.inertia_
# Plotting the curve with 'k'-value vs SSE
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster for MonetaryValue")
plt.ylabel("SSE")
# Save the Plot in current directory
plt.show()

# --Calculate R ,M and F groups--
# Create labels for Recency and Frequency
r_labels = range(6, 0,-1); f_labels = range(1, 7)
# Create labels for MonetaryValue
m_labels = range(1, 6)
# Assign these labels to 4 equal percentile groups
r_groups = pd.qcut(data_process['Recency'], q=6, labels=r_labels)
# Assign these labels to 4 equal percentile groups
f_groups = pd.qcut(data_process['Frequency'], q=6, labels=f_labels)
# Assign these labels to three equal percentile groups
m_groups = pd.qcut(data_process['MonetaryValue'], q=5, labels=m_labels)
# Create new columns R , F and M
data_process = data_process.assign(R = r_groups.values, F = f_groups.values,M = m_groups.values)
#print(data_process.head(50))

# Concat RFM quartile values to create RFM Segments
def join_rfm(x): return str(x['R']) + str(x['F']) + str(x['M'])
data_process['RFM_Segment_Concat'] = data_process.apply(join_rfm, axis=1)
rfm = data_process
#print(data_process['CustomerID'])

# Count num of unique segments
rfm_count_unique = rfm.groupby('RFM_Segment_Concat')['RFM_Segment_Concat'].nunique()
#print(rfm_count_unique.sum())

# Calculate RFM_Score
rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)
rfm = rfm.merge(transaction_df,on = 'customer_id',how ='inner')
rfm.drop(['Recency','Frequency','MonetaryValue', 'R', 'F', 'M', 'RFM_Segment_Concat','offer_transaction_date',
          'transaction_amount', 'points', 'accrual', 'redemption', 'customer_tier', 'offer_tier',
          'target_customer_age_segment', 'osd','id','oed'], 1, inplace=True)
print(rfm.columns.to_list())
rfm.to_csv("rfm_score_cust_offer.csv",index=False)

"""#Average RFM
rfm['Average Recency'] = rfm['Recency'].mean()
rfm['Average Frequency'] = rfm['Frequency'].mean()
rfm['Average MonetaryValue'] = rfm['MonetaryValue'].mean()"""

# Calculate average values for each RFM_Score, and return a size of each segment
"""rfm_level_agg = rfm.groupby('RFM_Score').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']}).round(1)

rfm_level_agg.columns = rfm_level_agg.columns.droplevel()
rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']"""

"""rfmjoinedData_df1 = rfm_level_agg.merge(rfm, on = 'RFM_Score',how ='inner')
rfmjoinedData_df1.to_csv("rfmData.csv")
print(rfmjoinedData_df1.head(10))"""