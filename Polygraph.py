#Polygraph

#importing the libraries
#http://pandas.pydata.org/
import pandas as pd

#Flak
from flask import Flask, jsonify, abort, request

#Regular Expressions
import re

#Natural language processing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

#Importing Data
#importing the dataset using pandas
def import_dataset(dataset_name, file_type, **kwargs):
	  dataset = getattr(pd, "read_{}".format(file_type))(dataset_name, sep = '\t', quoting = 3, error_bad_lines=False, header=None)
	  return dataset

#dataset_name = 
#dataset_type = 
dataset = import_dataset(dataset_name, dataset_type)

ps = PorterStemmer()
#Cleaning the text
def clean_text(raw_text):    
    cleaned_text = re.sub('[^a-zA-Z0-9]', ' ', raw_text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.split()
    cleaned_text = [ps.stem(word) for word in cleaned_text if not word in set(stopwords.words('english'))]
    cleaned_text = ' '.join(cleaned_text)
    
    return cleaned_text

corpus = []

for i in range(len(dataset)):
    cleaned_text = clean_text(dataset[0][i])    
    corpus.append(cleaned_text)
    
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values
    
#Step 5: Split into training/test sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

#Step 6: Fitting the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=25, random_state=3, criterion="entropy")
classifier.fit(x_train, y_train)

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
    
    corpus = []
    corpus.append(clean_text(response))
    print (corpus)
    sparse = cv.transform(corpus).toarray()     
    y_pred = classifier.predict(sparse) 
    return str(y_pred[0])

if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True, port=80)



#Visualising the results
#cm = confusion_matrix(y_test, classifier.predict(x_test))
    

