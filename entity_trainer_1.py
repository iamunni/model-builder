from __future__ import unicode_literals, print_function
import json
import ast
import random
import argparse
import os
import pandas as pd
from datetime import datetime

from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from settings import MODEL_PATH, DATA_PATH
# from usecases.models import TrainingSet
# from usecases.models import UseCase, ModelConfig, Entity
from evaluation import Evaluate
from db import MailbotDB, db_save

default_entities = ['PNR', 'AWB', 'FlightNo', 'Airline', 'Location', 'Dates',
                    'DateText', 'OnwardDate', 'OnwardDateDesc', 'ReturnDate',
                    'ReturnDateDesc', 'SourceAirport', 'DestinationAirport',
                    'City', 'Airport', 'Times', 'FlightClass']


def train(args, db):
    """
    Set up the pipeline and entity recognizer, and train the new entity.
    training data
    Note: If you're using an existing model, make sure to mix in examples of
    other entity types that spaCy correctly recognized before. Otherwise, your
    model might learn the new type, but "forget" what it previously knew.
    https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
    """

    n_iter = 1
    model = None
    usecase = args.usecase
    source = args.source
    threshold_value = args.threshold
    print("Building model for usecase: %s , Source: %s, Threshold: %s" % (usecase, source, threshold_value))
    base_dir = Path("{}/usecases/{}".format(MODEL_PATH, usecase))
    output_dir = "{}/usecases/{}/Model1".format(MODEL_PATH, usecase)

    if not base_dir.exists():
        os.makedirs("{}/usecases/{}".format(MODEL_PATH, usecase))

    output_dir = Path(output_dir)
    if output_dir.exists():
        model = output_dir

    training_set = []
    labels = []
    if source == 'csv':
        df = pd.read_csv(DATA_PATH+'/usecases_trainingset.csv')
        trainingset = df[df['use_case_id'] == usecase][['sentence', 'entities']].values.tolist()
    else:
        trainingset = db.query('select sentence, entities from usecases_trainingset where use_case_id=%s' % usecase)

    for data in trainingset:
        training_set.append((data[0], {"entities": ast.literal_eval(data[1])}))
        for label in ast.literal_eval(data[1]):
            labels.append(label[2])

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.load('en_core_web_sm')  # create blank Language class
        # nlp.vocab.vectors.name = 'spacy_pretrained_vectors'
        print("Created blank 'en' model")

    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    for label in labels:
        ner.add_label(label)

    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(training_set)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(training_set, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)

    # test the trained model
    # test_text = "Your PNR is U7223N."
    # doc = nlp(test_text)
    # print("Entities in '%s'" % test_text)
    # for ent in doc.ents:
    #     print(ent.label_, ent.text)

    # save model to output directory
    print('output_dir')

    if output_dir is not None:
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = 'usecase_{}.model'.format(usecase)
        nlp.to_disk(output_dir)
        # Getting the entity list labels
#         if source == 'csv':
#             entity_obj = pd.read_csv(DATA_PATH + '/usecases_entity.csv')[['name']].values.tolist()
#         else:
#             entity_obj = db.query('select name from usecases_entity where use_case_id=%s' % usecase)
#         # Entity.objects.filter(use_case=usecase).values()
#         entity_label_list = []
#         for e in entity_obj:
#             entity_label_list.append(e[0])

#         e = Evaluate()
#         beam_score = Evaluate.beam_score(nlp, training_set)
#         eval_score = Evaluate.score(nlp, training_set, entity_label_list)
#         if beam_score is not None:
#             model_full_details = e.basic_view(training_set, beam_score, threshold_value)
#             model_summary = model_full_details['summary']
#             model_matrix = model_full_details['matrix']
#             # model_matrix = json.loads(model_matrix)
#             model_efficiency = model_full_details['efficiency']
#             model_accuracy = model_full_details['precision']
#             model_error = model_full_details['recall']
#             model_threshold = threshold_value
#         else:
#             model_summary = None
#             model_matrix = None
#             model_efficiency = 0
#             model_accuracy = 0
#             model_error = 0
#             model_threshold = threshold_value
#         model_name = "Model1"
#         # Add/Update to DB model/create
#         if source == 'db':
#             db_save(db_conn, model_name, usecase, beam_score, eval_score, output_dir, model_summary, model_matrix,
#                     model_efficiency, model_accuracy['percentage'], model_error['percentage'], model_threshold)
#             print("Saved The Model Details!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model 1')
    parser.add_argument('usecase', metavar='U', type=int, help='A valid usecase ID')
    parser.add_argument('--source', metavar='S', type=str, default='db',
                        help='A valid souce of trainingset : csv or db')
    parser.add_argument('-threshold', metavar='--T', type=float, default=0.3,
                        help='A value from 0 to 1')
    args = parser.parse_args()
    db_conn = None
    if args.source == 'db':
        db_conn = MailbotDB()
    train(args, db_conn)
