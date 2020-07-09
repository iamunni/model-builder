import pickle
from pathlib import Path
import re
import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from settings import MODEL_PATH, DATA_PATH
from db import MailbotDB


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\n", "", string)
    string = re.sub(r"\r", "", string)
    string = re.sub(r"[0-9]", "digit", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def train(args, db):
    app = args.application

    if args.source == 'csv':
        df = pd.read_csv(DATA_PATH+'/usecases_trainingset.csv')
        training_set = df[['sentence', 'entities', 'use_case_id']].values.tolist()
    else:
        training_set = db.query('''select sentence, entities, use_case_id from usecases_trainingset 
        where use_case_id in (select id from usecases_usecase WHERE application_id=%s)''', app)

    training_set = [(i[0].lower().strip(), i[2]) for i in training_set]

    output_dir = "{}/classifiers/".format(MODEL_PATH)

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    # Adding the New Classifier Code into the BackEnd
    X = []
    y = []
    for i in range(len(training_set)):
        X.append(clean_str(training_set[i][0]))
        y.append(training_set[i][1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    # pipeline of feature engineering and model
    model = Pipeline([('vectorizer', CountVectorizer()),
                      ('clf', OneVsRestClassifier(MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)))])
    # fit model with training data
    model.fit(X_train, y_train)

    # cl = NaiveBayesClassifier(set(training_set))
    output_path = '{}/classifiers/app-{}-classifier.model'.format(MODEL_PATH, app)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model written out to {}".format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model 1')
    parser.add_argument('application', metavar='A', type=int, help='A valid usecase ID')
    parser.add_argument('--source', metavar='S', type=str, default='db',
                        help='A valid souce of trainingset : csv or db')
    args = parser.parse_args()
    db_conn = None
    if args.source == 'db':
        db_conn = MailbotDB()
    train(args, db_conn)
