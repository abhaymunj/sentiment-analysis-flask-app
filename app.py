import pandas as pd
from flask import Flask, render_template, request
from textblob import TextBlob
from sklearn.externals import joblib

# Initate flask application
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# Stopwords to remove
stopwords = ['and', 'this', 'that', 'are', 'us', 'we', 'he',
             'she', 'they', 'them', 'was', 'will', 'a',
             'i', 'the', 'am']

# For translate
deletedigits = str.maketrans(dict.fromkeys("1234567890"))
deletepunc = str.maketrans(dict.fromkeys("!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~÷"))

# Gets the data from the form on index.html
# Predicting Sentiment via NB Classifier


@app.route('/analyze_text', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        # Input for calculating subjectivity
        input_blob = TextBlob(input_text)
        # Convert input data string to list
        input_text_list = input_text.split()
        # Convert every list item to lowercase
        input_text_list = [x.lower() for x in input_text_list]
        # Remove numbers
        input_text_list = [x.translate(deletedigits) for x in input_text_list]
        # Remove punctuation
        input_text_list = [x.translate(deletepunc) for x in input_text_list]
        input_text_list = list(filter(None, input_text_list))
        # Remove stopwords
        input_text_list = [x for x in input_text_list if x not in(stopwords)]
        # Remove empty elements in the list
        input_text_list = list(filter(None, input_text_list))
        input_text_list = ' '.join(input_text_list)
        get_clean_text = [input_text_list]
        # Vectorize cleaned text input
        vectorized_input = count_vectorizer.transform(get_clean_text)
        # Make prediction
        sentiment_prediction = model.predict(vectorized_input)
        sentiment_prediction = int(sentiment_prediction)
        # Calculate the probabilities of the prediction
        sentiment_prediction_probability = model.predict_proba(
            vectorized_input)
        # Get the class with the maximum probability
        # Max with sort the list of probabilities for all classes
        # The probability at index 0 is the max
        sentiment_class_with_max_probability = max(
            sentiment_prediction_probability[0])
        # label the predicted class
        # 0 - Negative
        # 1 - Positive
        if sentiment_prediction == 0:
            sentiment_prediction_label = 'Negative'
            tag_color = '#ed6456'
        elif sentiment_prediction == 1:
            sentiment_prediction_label = 'Positive'
            tag_color = '#51bc7f'
        else:
            sentiment_prediction_label = 'Oops! The mystic has drunk himself to sleep!'

        # Round off to two decimal places
        sentiment_class_with_max_probability = round(
            sentiment_class_with_max_probability, 2)
        # Multiply by 100
        sentiment_class_with_max_probability = sentiment_class_with_max_probability * 100
        # Round off to two decimal places again
        sentiment_class_with_max_probability = round(
            sentiment_class_with_max_probability)
        # Convert to string and add % symbol
        sentiment_class_with_max_probability = str(
            sentiment_class_with_max_probability) + ' %'

        # Calculating subjectivity
        subjectivity_value = input_blob.subjectivity
        subjectivity_label = 'Oops! Somthing went wrong'
        if subjectivity_value < 0.5:
            subjectivity_label = 'Objective'
        else:
            subjectivity_label = 'Subjective'
        # Converting to percentage - looks better!
        subjectivity_value = "%.2f" % subjectivity_value
        subjectivity_value = float(subjectivity_value) * 100
        subjectivity_value = str(str(subjectivity_value) + ' %')

        return render_template('index.html', sentiment_temp_label=sentiment_prediction_label,
                               sentiment_temp_confidence=sentiment_class_with_max_probability, tag_color=tag_color,
                               subjectivity_level=subjectivity_value, subjectivity_label=subjectivity_label, input_text_template=input_text)


if __name__ == '__main__':
    # load Pickle objects for the model and vectorizer
    model = joblib.load('nb_clf.pkl')
    count_vectorizer = joblib.load('count_vect.pkl')
    # Switch debug to False in production
    app.run(debug=False)
