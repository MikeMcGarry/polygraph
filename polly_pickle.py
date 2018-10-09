'''
This is used to train your NLP model before pickling it and uploading it 
into an S3 bucket. Next step is to build in reinforcement learning and to deploy
as a lambda function instead of on an instance
''''

#importing the libraries
#http://pandas.pydata.org/
import pandas as pd

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
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

#Step 1: Importing Data
#importing the dataset using pandas
def import_dataset(dataset_name, file_type, **kwargs):
	  dataset = getattr(pd, "read_{}".format(file_type))(dataset_name, sep = '\t', quoting = 3, error_bad_lines=False, header=None)
	  return dataset

#Step 1: Importing Data
#The name of the file with your dataset goes here, the file should have the
#text in the first column and the lable in the second column      
dataset_name = ''
#The format of your dataset goes here
dataset_type = 'csv'
#Import the dataset
dataset = import_dataset(dataset_name, dataset_type)
#Create porter stemmer for stemming words
ps = PorterStemmer()
#Cleaning the text

def clean_text(raw_text):    
    #Ensure that all words have a single space between them
    cleaned_text = re.sub('[^a-zA-Z0-9]', ' ', raw_text)
    #Make all the text lower case
    cleaned_text = cleaned_text.lower()
    #Split the test into a list of words
    cleaned_text = cleaned_text.split()
    #Remove stopwords and call stem on all other words
    cleaned_text = [ps.stem(word) for word in cleaned_text if not word in set(stopwords.words('english'))]
    #Reconnect the text together
    cleaned_text = ' '.join(cleaned_text)
    #Return the cleaned text
    return cleaned_text

#Initialise the corpus
corpus = []

#Fill the corpus with cleaned text from the dataset
for i in range(len(dataset)):
    cleaned_text = clean_text(dataset[0][i])    
    corpus.append(cleaned_text)

#Reduce the number of features
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()

#Break off target label (second column in dataset)
y = dataset.iloc[:,1].values
    
#Step 5: Split into training/test sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

#Step 6: Fitting the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=25, random_state=3, criterion="entropy")
classifier.fit(x_train, y_train)

#Pickle the model for storage in S3 to be reloaded by the classifier
polygraphpipe = Pipeline([('cv', cv), ('classifier', classifier)])
joblib.dump(polygraphpipe, 'polygraphpipe.pkl', protocol = 0)

