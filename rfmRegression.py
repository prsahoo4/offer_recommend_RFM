# Import libraries
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import squarify
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Read dataset
transaction_df = pd.read_excel('Sample_Data_Offer_Customer_Transaction\Offer_Transaction_Log.xlsx')
transaction_df.to_csv("transaction_df.csv")
transaction_df.drop(['OfferTargetAge'], 1, inplace=True)

cust_df = pd.read_excel("Sample_Data_Offer_Customer_Transaction\Customer_Master.xlsx")
cust_df.to_csv("cust_df.csv")
cust_df.drop(['CustomerTitle','CustomerMaritalStatus','CustomerAgeSegment','CustomerAge','CustomerTier'], 1,
             inplace=True)

offer_df = pd.read_excel("Sample_Data_Offer_Customer_Transaction\Offer_Master.xlsx")
offer_df.to_csv("offer_df.csv")
offer_df.drop(['Targeted Marital Status','Targeted Tier','Targeted Gender','Targeted Age'],1,inplace=True)

#joining the data
joinedData_df = cust_df.merge(transaction_df, on = 'CustomerID',how ='inner')
joinedData_df.to_csv("Join.csv" , index=False)

def Rename_OfferTier(df):
    for col in ['CustomerTier']:
        df[col] = df[col].map({'B':'Bronze', 'S':'Saphire','P':'Platinum','G':'Gold'})
    return df

joinedData_df = Rename_OfferTier(joinedData_df)
offerTransaction = offer_df.merge(joinedData_df, on='OfferId',how='inner')
offerTransaction.to_excel("DataFull.xlsx")
# Convert InvoiceDate from object to datetime format
offerTransaction['InvoiceDate'] = pd.to_datetime(offerTransaction['OfferTransactionDate'])

# Drop NA values from online
offerTransaction.dropna()

# --Group data by customerID--
# Create snapshot date
snapshot_date = offerTransaction['InvoiceDate'].max() + timedelta(days=1)
print(snapshot_date)

# Grouping by CustomerID
data_process = offerTransaction.groupby(['CustomerID'],as_index=False).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'OfferId': 'count',
        'TransactionAmount': 'sum'})
print(data_process.columns.tolist())
# Rename the columns
data_process.rename(columns={'InvoiceDate': 'Recency',
                         'OfferId': 'Frequency',
                         'TransactionAmount': 'MonetaryValue'}, inplace=True)

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
#plt.show()

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
#plt.show()

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
#plt.show()

# --Calculate R ,M and F groups--
# Create labels for Recency and Frequency
r_labels = range(4, 0,-1); f_labels = range(1, 6)
# Create labels for MonetaryValue
m_labels = range(1, 6)
# Assign these labels to 4 equal percentile groups
r_groups = pd.qcut(data_process['Recency'], q=4, labels=r_labels)
# Assign these labels to 4 equal percentile groups
f_groups = pd.qcut(data_process['Frequency'], q=5, labels=f_labels)
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


# Calculate average values for each RFM_Score, and return a size of each segment
rfm_level_agg = rfm.groupby('RFM_Score').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']}).round(1)

rfm_level_agg.columns = rfm_level_agg.columns.droplevel()
rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']

rfmjoinedData_df1 = rfm_level_agg.merge(rfm, on = 'RFM_Score',how ='inner')



#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes=rfm_level_agg['Count'],
              label=['Score :14',
                     'Score :13',
                     'Score :12',
                     'Score :11',
                     'Score :10','Score :9','Score :8','Score :7','Score :6','Score :5','Score :4',
                     'Score :3'], alpha=.6 )
plt.title("RFM Segments",fontsize=18,fontweight="bold")
plt.axis('off')
#plt.show()

rfmTransactionData = rfm.merge(offerTransaction, on='CustomerID',how='inner')
print(rfmTransactionData.head(10))
rfmTransactionData.to_csv('RFMTRANSDATA.csv')

rfm_TargetSpace = rfmTransactionData.groupby('RFM_Score').agg({
    'CustomerTier': pd.Series.mode,
    'CustomerGender':pd.Series.mode,'CustomerCountry':pd.Series.mode,'OfferTier':pd.Series.mode,
    'CustomerAge':pd.Series.mode})

rfm_TargetSpace.rename(columns={'CustomerTier': 'targetedCustomerTier',
                         'CustomerGender': 'targetedCustomerGender',
                         'CustomerCountry': 'targetedCustomerCountry','OfferTier':'targetedOfferTier',
                                'CustomerAge':'targetedAge'}, inplace=True)

print(rfm_TargetSpace.head(14))

rfmjoinedData_df2 = rfm_TargetSpace.merge(rfmjoinedData_df1, on = 'RFM_Score',how ='inner')
rfmjoinedData_df2.to_csv('CustOfferRFM.csv')