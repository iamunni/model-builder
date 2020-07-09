from __future__ import unicode_literals, print_function
import argparse
import ast
import numpy
import spacy
import pandas as pd
from pathlib import Path
from spacy.util import minibatch, compounding
from settings import MODEL_PATH, DATA_PATH
from evaluation import Evaluate
from db import MailbotDB, db_save


def train(args, db_conn):
    model = None
    usecase = args.usecase
    source = args.source
    print("Building model for usecase: %s , Source: %s" % (usecase, source))

    output_dir = "{}/usecases/{}/Model2".format(MODEL_PATH, usecase)

    output_dir = Path(output_dir)
    if output_dir.exists():
        model = output_dir

    training_data = []
    labels = []

    if source == 'csv':
        trainingset = pd.read_csv(DATA_PATH + '/usecases_trainingset.csv')[['sentence', 'entities']].values.tolist()
    else:
        trainingset = db_conn.query(
            'select sentence, entities from usecases_trainingset where use_case_id=%s' % usecase)

    for data in trainingset:
        training_data.append((data[0], {"entities": ast.literal_eval(data[1])}))
        for label in ast.literal_eval(data[1]):
            # if label[2] not in labels and label[2] != 'EndDate':
            labels.append(label[2])

    # CHECK THE 'training_data' Format
    # training_data = [('what is the price of polo?', {'entities': [(21, 25, 'PrdName')]}), ('what is the price of ball?', {'entities': [(21, 25, 'PrdName')]})]

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        # nlp = spacy.load('en_core_web_sm')  # create blank Language class
        nlp = spacy.load('en_core_web_lg')

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

    # if model is None:
    #    optimizer = nlp.begin_training()
    # else:
    #    # Note that 'begin_training' initializes the models, so it'll zero out
    #    # existing entity types.
    #    optimizer = nlp.entity.create_optimizer()

    n_iter = 1

    Loops_Lowest_loss = 10000
    Loops_Lowest_alpha = 10000
    Loops_Lowest_LR = 10000
    Loops_Lowest_Drop = 10000
    Best_Iteration = 0
    Loops_Lowest_evaluation_score = {}
    Loops_Lowest_beam_score = {}

    for inum in range(n_iter):
        print("ITERATION   >>> ", inum)

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

        Lowest_loss = 10000
        Lowest_i = 10000
        Lowest_opt = 10000
        Lowest_alpha = 10000
        Lowest_LR = 10000
        Lowest_Drop = 10000

        is_break = False

        Lowest_evaluation_score = {}
        Lowest_beam_score = {}

        all_Anotations = []
        all_Anotations.clear()

        # optimizer = nlp.begin_training()
        # optimizer = nlp.entity.create_optimizer()

        for iLearnRate in numpy.arange(0.001, 0.002, 0.001):
            # print("IN-2",iLearnRate)
            # optimizer.learn_rate = iLearnRate

            for ialpha in numpy.arange(0.001, 0.1, 0.01):
                # print("IN",ialpha)
                # optimizer.alpha = ialpha

                for i in numpy.arange(0.1, 0.9, 0.05):
                    # print("IN-3",i)
                    # ============ <=alpha=> 0.001 <=LearnRate=> 0.09099999999999998 <=drop=> 0.1 ============
                    print("\n\n============", "ITERATION   >>> ", inum, "<=LearnRate=>", iLearnRate, "<=alpha=>",
                          ialpha, "<=drop=>", i, "============")
                    # random.shuffle(training_data)
                    losses = {}
                    cfg = {}

                    # if model is None:
                    #    optimizer = nlp.begin_training()
                    # else:
                    #    optimizer = nlp.entity.create_optimizer()

                    optimizer = nlp.begin_training()
                    optimizer.beta1 = 0.9
                    optimizer.learn_rate = iLearnRate
                    optimizer.alpha = ialpha

                    # batches = minibatch(training_data, size=compounding(4., 32., 1.001))
                    batches = minibatch(training_data)
                    with nlp.disable_pipes(*other_pipes):
                        for batch in batches:
                            # print(batch)
                            texts, annotations = zip(*batch)
                            nlp.update(texts, annotations, sgd=optimizer, drop=i, losses=losses)

                    print('losses => ', losses['ner'])
                    print('\n\n', "Lowest Iteration Loose >> ", Loops_Lowest_loss, '\t Lowest_loss', Lowest_loss,
                          '<==>', 'Losses', losses['ner'], '\n\n')

                    if Lowest_loss > losses['ner']:
                        print('\t\t evaluate.score')
                        evaluation_score = Evaluate.score(nlp, training_data, labels)
                        print('\n', evaluation_score)

                        print('\t\t evaluate.BeamScore')
                        beam_score = Evaluate.beam_score(nlp, training_data)
                        print('\n', beam_score)

                        Lowest_loss = losses['ner']
                        Lowest_evaluation_score = evaluation_score
                        Lowest_beam_score = beam_score

                        Lowest_alpha = ialpha
                        Lowest_LR = iLearnRate
                        Lowest_Drop = i

                        print(">>>>>>>>>>>>>> inside Lowest Value <<<<<<<<<<<<<<<")
                        print(Lowest_loss, Loops_Lowest_loss)
                        if Lowest_loss < Loops_Lowest_loss:
                            print(">>>>>>>>>>>>>> inside Lowest Loop Value <<<<<<<<<<<<<<<")

                            Loops_Lowest_loss = Lowest_loss
                            Best_Iteration = inum
                            Loops_Lowest_alpha = Lowest_alpha
                            Loops_Lowest_LR = Lowest_LR
                            Loops_Lowest_Drop = Lowest_Drop
                            Loops_Lowest_evaluation_score = Lowest_evaluation_score
                            Loops_Lowest_beam_score = Lowest_beam_score

                            if output_dir is not None:
                                if not output_dir.exists():
                                    output_dir.mkdir()

                            nlp.meta['name'] = 'usecase_{}.model'.format(usecase)
                            nlp.to_disk(output_dir)
                            print("Saved model to", output_dir)

                # END of drop Loop

            # END of alpha Loop

        # END of learn_rate Loop

        print("\t==================================================================================================")
        print("\t==================================================================================================")
        print("\t<=alpha=>", Lowest_alpha, "<=LearnRate=>", Lowest_LR, "<=drop=>", Lowest_Drop)
        print('\tLowest_loss : ', Lowest_loss)
        print(Lowest_evaluation_score)
        print(Lowest_beam_score)

    # END of ITERATION Loop

    print("===================================ITERATION===============================================================")
    print("==================================================================================================")
    print("Best Iretation index : ", Best_Iteration, "<=alpha=>", Loops_Lowest_alpha, "<=LearnRate=>", Loops_Lowest_LR,
          "<=drop=>", Loops_Lowest_Drop)
    print('Lowest_loss in all LOOP : ', Loops_Lowest_loss)
    print(Loops_Lowest_evaluation_score)
    print(Loops_Lowest_beam_score)

    if source == 'csv':
        entity_obj = pd.read_csv(DATA_PATH + '/usecases_entity.csv')[['name']].values.tolist()
    else:
        entity_obj = db_conn.query('select name from usecases_entity where use_case_id=%s' % usecase)

    entity_label_list = []
    for e in entity_obj:
        entity_label_list.append(e[0])

    # -------- Calling Functions for evaluating ----------
    beam_score = Evaluate.beam_score(nlp, training_data)
    eval_score = Evaluate.score(nlp, training_data, entity_label_list)
    model_name = "Model2"
    # Add/Update to DB model/create
    if source == 'db':
        db_save(model_name, usecase, beam_score, eval_score, output_dir)
        print("Saved The Model Details!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model 2')
    parser.add_argument('usecase', metavar='U', type=int, help='A valid usecase ID')
    parser.add_argument('--source', metavar='S', type=str, default='db',
                        help='A valid souce of trainingset : csv or db')
    args = parser.parse_args()
    db_conn = None
    if args.source == 'db':
        db_conn = MailbotDB()
    train(args, db_conn)
