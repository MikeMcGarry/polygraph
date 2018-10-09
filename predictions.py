'''
File to be deployed and run with nohup on an AWS Ubuntu instance with a Conda
Environment
'''

from sklearn.externals import joblib
from boto.s3.key import Key
from boto.s3.connection import S3Connection, Location
import boto
from flask import Flask, jsonify, abort, request
from nltk.stem.porter import PorterStemmer
import re
import urllib

#The name of the S3 bucket where your pickled model file is
BUCKET_NAME = ''
#The name of the pickled model file inside the S3 bucket
MODEL_FILE_NAME = ''
#The path on the local instance to save the file to
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_NAME
#Raw stopwords to remove, in raw form as problematic to download on AWS instance
stopwords_raw = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                 'against', 'between', 'into', 'through', 'during', 'before',
                 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
                 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
                 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
                 'won', 'wouldn']

#App to handle the API requests
app = Flask(__name__)

#Check that the API is live
@app.route('/')
def index():
    return "We are live"
    
#Returns the prediction from a GET request with the email body 
@app.route('/polygraph', methods=['GET'])
def result(*args):
    #Collect the arguments from the GET request
    arguments = request.args
    
    try:
        #Try and collect the response argument
        response = str(arguments['response'])
    except:
        #If it does not exist abort
        abort(404)    
    
    #predict the label based on the email response
    y_pred = predict(response)
    
    #return the prediction
    return str(y_pred[0])


#Cleaning the text
def clean_text(raw_text): 
    #Create porter stemmer for stemming words
    ps = PorterStemmer()
    #Ensure that all words have a single space between them
    cleaned_text = re.sub('[^a-zA-Z0-9]', ' ', raw_text)
    #Make all the text lower case
    cleaned_text = cleaned_text.lower()
    #Split the test into a list of words
    cleaned_text = cleaned_text.split()
    #Remove stopwords and call stem on all other words
    cleaned_text = [ps.stem(word) for word in cleaned_text if not word in set(stopwords_raw)]
    #Reconnect the text together
    cleaned_text = ' '.join(cleaned_text)
    #Return the cleaned text
    return cleaned_text

#Loads the pickled NLP model
def load_model():
    #open an S3 connection
    conn = boto.connect_S3()
    #connect to the bucket with our pickled model
    bucket = conn.get_bucket(BUCKET_NAME)
    #create a key object
    key_obj = Key(bucket)
    #connect to the pickled model file
    key_obj.key = MODEL_FILE_NAME
    #save model to local filesystem
    contents = key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)
    #load and return the pickled model
    return joblib.load(MODEL_LOCAL_PATH)

#Predicts the label for the email
def predict(data):
  final_formatted_data = []
  #Decode the URL encoded response
  response_decoded = urllib.unquote_plus(data)
  #Clean the response
  final_formatted_data.append(clean_text(response_decoded))
  #Return the prediction using the loaded model
  return load_model().predict(final_formatted_data)