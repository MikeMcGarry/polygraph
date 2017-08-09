from sklearn.externals import joblib
from boto.s3.key import Key
from boto.s3.connection import S3Connection, Location
import boto
from flask import Flask, jsonify, abort, request
from nltk.stem.porter import PorterStemmer
import re
import urllib
BUCKET_NAME = ''
MODEL_FILE_NAME = ''
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_NAME
stopwords_raw = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

app = Flask(__name__)

@app.route('/')
def index():
    return "We are live"
    
@app.route('/polygraph', methods=['GET'])
def result(*args):
    arguments = request.args
    
    try:
        response = str(arguments['response'])
    except:
        abort(404)    

    y_pred = predict(response)
    
    
    return str(y_pred[0])


#Cleaning the text
def clean_text(raw_text): 
    ps = PorterStemmer()
    cleaned_text = re.sub('[^a-zA-Z0-9]', ' ', raw_text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.split()
    cleaned_text = [ps.stem(word) for word in cleaned_text if not word in set(stopwords_raw)]
    cleaned_text = ' '.join(cleaned_text)
    
    return cleaned_text

def load_model():
  conn = boto.connect_S3()
  bucket = conn.get_bucket(BUCKET_NAME)
  key_obj = Key(bucket)
  key_obj.key = MODEL_FILE_NAME

  contents = key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)
  return joblib.load(MODEL_LOCAL_PATH)

def predict(data):
  final_formatted_data = []
  response_decoded = urllib.unquote_plus(data)
  final_formatted_data.append(clean_text(response_decoded))
  return load_model().predict(final_formatted_data)