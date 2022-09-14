import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel




input1 = 'O1111'
input2 = '15% Discount on your next Walmart Store Visit'



#taking inputs and converting it into existing dataframe(metadata)
metadata = pd.read_excel('Offer_Master - Copy.xlsx')
df2 = pd.DataFrame({"OfferId":[input1],
                    "OfferContent":[input2]})
metadata = metadata.append(df2,ignore_index=True)

#filling NaN with zero for offercontent
metadata['OfferContent'] = metadata['OfferContent'].fillna('')

#Vectorizing
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(metadata['OfferContent'])

#cosine similarity
cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix )

#Construct a reverse map of indices and offer titles
indices = pd.Series(metadata.index, index=metadata['OfferId']).drop_duplicates()

# Function that takes in offer title as input and outputs most similar offers/properties
def get_recommendations(title, cosine_sim=cosine_sim):

    # Get the index of the offer that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all offers with that offer
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the offers based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar offers
    sim_scores = sim_scores[0:5]

    # Get the offer indices
    offer_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar offers
    return metadata['OfferId'].iloc[offer_indices]


print(get_recommendations(input1))