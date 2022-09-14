# Import libraries
import pandas as pd
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import squarify
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


# Read dataset
transaction_df = pd.read_excel('Sample_Data_Offer_Customer_Transaction\Offer_Transaction_Log.xlsx')
transaction_df.to_csv("Offer_Transaction_Log.csv")


cust_df = pd.read_excel("Sample_Data_Offer_Customer_Transaction\Customer_Master.xlsx")
cust_df.to_csv("Customer_Master.csv")
cust_df.drop(['CustomerTitle','CustomerMaritalStatus','CustomerAge','CustomerTier'], 1,
             inplace=True)


offer_df = pd.read_excel("Sample_Data_Offer_Customer_Transaction\Offer_Master.xlsx")
offer_df.to_csv("Offer_Master.csv")
offer_df.drop(['TargetedMaritalStatus','TargetedTier','TargetedGender','TargetedAge'],1,inplace=True)

#joining the data
joinedData_df = cust_df.merge(transaction_df, on = 'CustomerID',how ='inner')
joinedData_df.to_csv("Join.csv" , index=False)

def Rename_OfferTier(df):
    for col in ['CustomerTier']:
        df[col] = df[col].map({'B':'Bronze', 'S':'Saphire','P':'Platinum','G':'Gold'})
    return df

joinedData_df = Rename_OfferTier(joinedData_df)
offerTransaction = offer_df.merge(joinedData_df, on='OfferId',how='outer')


# Drop NA values from online (#### NOT DROPPING IN THIS)
offerTransaction['CustomerID'].fillna(0,inplace = True)
offerTransaction['TransactionAmount'].fillna(0,inplace = True)
offerTransaction['OfferTransactionDate'].fillna(datetime(2018,1,1),inplace = True)

# Convert InvoiceDate from object to datetime format
offerTransaction['InvoiceDate'] = pd.to_datetime(offerTransaction['OfferTransactionDate'])

offerTransaction.to_excel("DataFull.xlsx")

# --Group data by customerID--
# Create snapshot date
snapshot_date = offerTransaction['InvoiceDate'].max() + timedelta(days=1)

# Grouping by CustomerID
data_process = offerTransaction.groupby(['OfferId'],as_index=False).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'CustomerID': 'count',
        'TransactionAmount': 'sum'})

# Rename the columns
data_process.rename(columns={'InvoiceDate': 'Recency',
                         'CustomerID': 'Frequency',
                         'TransactionAmount': 'MonetaryValue'}, inplace=True)

# Print top rows and shape of dataframe

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
r_labels = range(2, 0,-1); f_labels = range(1, 3)
# Create labels for MonetaryValue
m_labels = range(1, 4)
# Assign these labels to 4 equal percentile groups
r_groups = pd.qcut(data_process['Recency'], q=3, duplicates='drop',labels=r_labels)
# Assign these labels to 4 equal percentile groups
f_groups = pd.qcut(data_process['Frequency'], q=4, duplicates='drop',labels=f_labels)
# Assign these labels to three equal percentile groups
m_groups = pd.qcut(data_process['MonetaryValue'], q=5,duplicates='drop', labels=m_labels)
# Create new columns R , F and M
data_process = data_process.assign(R = r_groups.values, F = f_groups.values,M = m_groups.values)

cust_df = pd.read_excel("Sample_Data_Offer_Customer_Transaction\Customer_Master.xlsx")
cust_df = Rename_OfferTier(cust_df)

targeted = cust_df.groupby(['CustomerAgeSegment'],as_index=False).agg({
        'CustomerID': 'count'})
targeted.rename(columns={'CustomerID': 'targeted'}, inplace=True)
cust_Allsegment = pd.DataFrame({'CustomerAgeSegment':["All"],'targeted':[cust_df.shape[0]]})
targeted = targeted.append(cust_Allsegment, ignore_index = True)
targetedTier = cust_df.groupby(['CustomerTier'],as_index=False).agg({
        'CustomerID': 'count'})
targetedTier.rename(columns={'CustomerID': 'targetedTier'}, inplace=True)

availed = offerTransaction.groupby(['OfferTargetAge','OfferId'],as_index=False).agg({
        'CustomerID': 'count'})
availed.rename(columns={'CustomerID': 'availed','OfferTargetAge':'CustomerAgeSegment'}, inplace=True)
availedTier = offerTransaction.groupby(['OfferTier','OfferId'],as_index=False).agg({
        'CustomerID': 'count'})
availedTier.rename(columns={'CustomerID': 'availedTier','OfferTier':'CustomerTier'}, inplace=True)

joinedTarAva1_df = targeted.merge(availed, on = 'CustomerAgeSegment',how ='inner')
joinedTarAva2_df = targetedTier.merge(availedTier, on = 'CustomerTier',how ='inner')


def conversionRate(x,y):
    return x*100/y;

joinedTarAva1_df['conversion_rate'] = joinedTarAva1_df.apply(lambda x: conversionRate(x.availed, x.targeted), axis=1)
joinedTarAva2_df['conversion_rateTier'] = joinedTarAva2_df.apply(lambda x: conversionRate(x.availedTier, x.targetedTier), axis=1)

mapping = dict(offer_df[['OfferId', 'OfferName']].values)
data_process['OfferName'] = data_process.OfferId.map(mapping)
mapping = dict(offer_df[['OfferId', 'OfferCategory']].values)
data_process['OfferCategory'] = data_process.OfferId.map(mapping)
mapping = dict(offer_df[['OfferId', 'ProductCode']].values)
data_process['ProductCode'] = data_process.OfferId.map(mapping)
mapping = dict(offer_df[['OfferId', 'ProductName']].values)
data_process['ProductName'] = data_process.OfferId.map(mapping)
mapping = dict(offer_df[['OfferId', 'PartnerCode']].values)
data_process['PartnerCode'] = data_process.OfferId.map(mapping)
mapping = dict(offer_df[['OfferId', 'PartnerName']].values)
data_process['PartnerName'] = data_process.OfferId.map(mapping)
mapping = dict(offer_df[['OfferId', 'DiscountPercentage']].values)
data_process['DiscountPercentage'] = data_process.OfferId.map(mapping)
mapping = dict(offer_df[['OfferId', 'StartDate']].values)
data_process['StartDate'] = data_process.OfferId.map(mapping)
mapping = dict(offer_df[['OfferId', 'EndDate']].values)
data_process['EndDate'] = data_process.OfferId.map(mapping)

data_process_final = data_process.merge(joinedTarAva1_df, on = 'OfferId',how ='outer')
data_process_final = data_process_final.merge(joinedTarAva2_df, on = 'OfferId',how ='outer')
print(data_process_final.head)
data_process_final.to_excel("data_process_final.xlsx")
# Calculate RFM_Score
data_process_final['RFM_Score'] = data_process_final[['R','F','M']].sum(axis=1)
data_process_final.drop(['R','F','M'], 1,
             inplace=True)

data_process_final['quarterStart'] = data_process_final.StartDate.dt.quarter
data_process_final['quarterEnd'] = data_process_final.EndDate.dt.quarter
data_process_final.to_excel("data_process_final.xlsx")
discountOfferId = data_process_final[data_process_final.OfferCategory.eq("Discounts")]

partner_code = "AM"
product_code = "P006"

partner_df = discountOfferId[discountOfferId["PartnerCode"]==partner_code]
product_df = partner_df[partner_df["ProductCode"]==product_code]
record = product_df[product_df["MonetaryValue"] == product_df["MonetaryValue"].max()]
print(record["DiscountPercentage"])

"""update = pd.read_excel('Sample_Data_Offer_Customer_Transaction\Offer_Master.xlsx')

update.to_csv('Sample_Data_Offer_Customer_Transaction\offer_data.csv')"""

"""temp = discountOfferId.groupby(['ProductCode','RFM_Score'],as_index=False).head()
temp.to_excel("temp.xlsx")"""