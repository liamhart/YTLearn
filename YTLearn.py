# -*- coding: utf-8 -*-
"""
Spyder Editor

@author Liam D. Hart
University of Rhode Island, B.S. Computer Science

Boiler plate code source: https://developers.google.com/youtube/v3/quickstart/python

"""

# Sample Python code for user authorization

import httplib2
import sys

from apiclient.discovery import build
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret.
CLIENT_SECRETS_FILE = "client_secret.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
YOUTUBE_READ_WRITE_SSL_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

# Key to access YouTube API
api_key=<API_KEY>

# This variable defines a message to display if the CLIENT_SECRETS_FILE is
# missing.
MISSING_CLIENT_SECRETS_MESSAGE = "WARNING: Please configure OAuth 2.0"

# Authorize the request and store authorization credentials.
def get_authenticated_service(args):
  flow = flow_from_clientsecrets(CLIENT_SECRETS_FILE, scope=YOUTUBE_READ_WRITE_SSL_SCOPE,
    message=MISSING_CLIENT_SECRETS_MESSAGE)

  storage = Storage("%s-oauth2.json" % sys.argv[0])
  credentials = storage.get()

  if credentials is None or credentials.invalid:
    credentials = run_flow(flow, storage, args)

  # Trusted testers can download this discovery document from the developers page
  # and it should be in the same directory with the code.
  return build(API_SERVICE_NAME, API_VERSION,
      http=credentials.authorize(httplib2.Http()))

args = argparser.parse_args()
service = get_authenticated_service(args)

### END BOILERPLATE CODE

def get_videos_by_playlist(service, playListID):
  uploads_list_id = playListID

  print("Videos in list %s" % uploads_list_id)

  # Retrieve the list of videos uploaded to the authenticated user's channel.
  playlistitems_list_request = service.playlistItems().list(
    playlistId=uploads_list_id,
    part="snippet",
    maxResults=50
  )

  while playlistitems_list_request:
    playlistitems_list_response = playlistitems_list_request.execute()
    
    # Generate list of items in the playlist
    playlist_item_id_list = []
    
    # Print information about each video.
    for playlist_item in playlistitems_list_response["items"]:
      #title = playlist_item["snippet"]["title"]
      video_id = playlist_item["snippet"]["resourceId"]["videoId"]
      # Append id to list
      playlist_item_id_list.append(video_id)
      #print("%s (%s)" % (title, video_id))
    print(playlist_item_id_list)

    playlistitems_list_request = service.playlistItems().list_next(
      playlistitems_list_request, playlistitems_list_response)
    
    return playlist_item_id_list


# Import requests for handling json and category of videos
import requests
import pandas as pd

def get_data(key, region, *ids):
    url = "https://www.googleapis.com/youtube/v3/videos?part=snippet&id={ids}&key={api_key}"
    r = requests.get(url.format(ids=",".join(ids), api_key=api_key))
    js = r.json()
    items = js["items"]
    cat_js = requests.get("https://www.googleapis.com/youtube/v3/videoCategories?part=snippet&regionCode={}&key={}".format(region,
        key)).json()
    name_and_cat_list = []
    categories = {d['id']: d["snippet"]['title'] for d in cat_js["items"]}
    for item in items:
        name_and_cat_list.append((item["snippet"]["title"], categories[item["snippet"]["categoryId"]]))
    return name_and_cat_list


def main():
    print("########################################################################################")
    print("Welcome to the 2017 version of YTLearn!")
    print("Instuctions: Please enter a valid YouTube playlist ID")
    print("             The program will parse the videos in said playlist")
    print("             Then generate a dataframe of titles and categories")
    print("             And test how well keywords predict category.")
    print("Example:     A playlist link will have some parts to it, for instance,")
    print("             a classical music playlist link will appear as follows-")
    print("             https://www.youtube.com/watch?v=mu_C_g8VoPE&list=PLDF5C30203C921976")
    print("             After the 'list' tag is the playlist ID, in this case, 'PLDF5C30203C921976'")
    print("########################################################################################")
    
    # Obtain user input
    playlist_id = input("Please enter a valid ID: ")
    
    
    # Take a sample by random playlist
    playlist_item_id_list = get_videos_by_playlist(service, playlist_id)
    
    # May take some time to obtain these lists given internet connection must be stable and O(n2) complexity
    arr = [[0 for j in range(2)] for i in range(len(playlist_item_id_list))]
    
    for i in range(len(playlist_item_id_list)):
        # Obtain data for analysis
        for title, cat in get_data(api_key, "IE", playlist_item_id_list[i]):
            arr[i][0] = title
            arr[i][1] = cat
    
    # Prepare data
    df = pd.DataFrame(data=arr,columns=['Titles','Categories'])
    X = df['Titles']
    y = df['Categories']
    
    # Support for unavailable videos
    for i in range(X.size):
        if(X[i] == 0) :
            X[i] = "None"
            y[i] = "None"
    
    print(df)
    
    # For modelling and natural lang processing
    from sklearn.model_selection import cross_val_score
    from re import sub
    from nltk.stem import PorterStemmer
    
    
    stemmer = PorterStemmer()
    
    print("######### Preparing data #########")
    new_data = []
    for i in range(X.size):
            new_data.append(sub("[^a-zA-Z]"," ",X[i]))
    
    lowercase_data = []
    for i in range(len(new_data)):
        lowercase_data.append(new_data[i].lower())
        
    stemmed_data = []
    for i in range(len(lowercase_data)):
        words = lowercase_data[i].split()
        stemmed_words = []
        for w in words:
            stemmed_words.append(stemmer.stem(w))
        stemmed_data.append("".join(stemmed_words))
    
    
    print("######### Setting up vector model #########")
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer="word",min_df=1,stop_words='english')
    docarray = vectorizer.fit_transform(stemmed_data).toarray()
    
    print("######### Model and XV #########")
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=.01)
    
    scores = cross_val_score(model,docarray,y,cv=10)
    print("Fold Accuracies: {}".format(scores))
    print("XV Accuracy: {:6.2f}".format(scores.mean()*100))

main()


