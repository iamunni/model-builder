import spacy
import json
from spacy.scorer import Scorer
from spacy.gold import GoldParse

from collections import defaultdict
import numpy as np
import pandas as pd


# global str


class Evaluate:
    @classmethod
    def score(cls, nlp, texts, labels):
        try:
            scorer = Scorer()
            for input_, annot in texts:
                text_entities = [entity for entity in annot.get('entities') if entity[2] in labels]
                doc_gold_text = nlp.make_doc(input_)
                gold = GoldParse(doc_gold_text, entities=text_entities)
                pred_value = nlp(input_)
                scorer.score(pred_value, gold)
                # print(scorer.scores)
            return scorer.scores
        except Exception as e:
            return ""
            # raise
            # sys.exit(1)

    @classmethod
    def beam_score(cls, nlp, texts):
        try:
            ret_data = []

            for input_, annot in texts:

                doc = nlp(input_)
                score_arr = []

                ner = nlp.get_pipe('ner')

                beams = nlp.entity.beam_parse([doc], beam_width=16, beam_density=0.0001)
                for score, ents in ner.moves.get_beam_parses(beams[0]):
                    ent_loc = []
                    ret_entities = []
                    entity_scores = defaultdict(float)
                    for start, end, label in ents:
                        entity_scores[(doc[start:end], label)] += score
                        doc_string = str(doc)
                        doc_string[doc[start:end].start_char:doc[start:end].end_char]
                        loc = [doc[start:end].start_char, doc[start:end].end_char, label]
                        ent_loc.append(loc)

                    # print('Score : ', score)

                    if entity_scores:
                        for x, y in entity_scores.items():
                            xx = (str(x[0]), x[1])
                            ret_entities.append(xx)

                    arr_score = (score, ret_entities, ent_loc)
                    score_arr.append(arr_score)

                data = (input_, score_arr)
                ret_data.append(data)

            return ret_data

        except Exception as e:
            return ""

    def models(self, nlp, training_data, labels):
        """
        :param nlp: Spacy model object
        :param training_data:  Validation Tests with Annotations
        :param labels:  Entities list used in the training

        :return: dictionary containing evaluation_score and beam_score
        """

        # print('evaluate.score')
        evaluation_score = self.score(nlp, training_data, labels)
        # print('\t\t', evaluation_score)

        # print('evaluate.beam_score')
        beam_score = self.beam_score(nlp, training_data)
        # print('\t\t', beam_score)
        ret_data = {'evaluation_score': evaluation_score, 'beam_score': beam_score}
        return ret_data

    def basic_view(self, anotated_doc, predicted_doc, ithreshold):
        """
                    Summary of whole Validation Emails
        """

        eval_data = self.evaluate(anotated_doc, predicted_doc, ithreshold)
        anot_dic = eval_data['annotated']
        pred_dict = eval_data['predicted']
        full_predict = eval_data['full_predict']

        summary = self.basic_summary(anot_dic, pred_dict)
        matrix = self.basic_matrix(anot_dic, full_predict)
        pre = self.calculate_pre(anot_dic, pred_dict, full_predict, ithreshold)

        precision = pre['precision']
        recall = pre['recall']
        efficiency = pre['efficiency']

        basic_data = dict()
        basic_data["precision"] = precision
        basic_data["recall"] = recall
        basic_data["efficiency"] = efficiency
        basic_data["summary"] = summary
        basic_data["matrix"] = matrix

        return basic_data

    def calculate_pre(self, anot_dic, pred_dict, full_predict_dict, ithreshold=0.10):
        ret_data = {}

        annotated_ents = []
        aarr = []
        for d in anot_dic:
            arr = [d.split(':-:')[1], d.split(':-:')[0], anot_dic[d],
                   d.split(':-:')[2], d.split(':-:')[3]]
            aarr.append(arr)
            annotated_ents.append(d.split(':-:')[1])

        annotated_ents = list(set(annotated_ents))

        parr = []
        for d in pred_dict:
            arr = [d.split(':-:')[1], d.split(':-:')[0], pred_dict[d],
                   d.split(':-:')[2], d.split(':-:')[3]]
            parr.append(arr)

        parr_full = []
        for d in full_predict_dict:
            arr = [d.split(':-:')[1], d.split(':-:')[0], full_predict_dict[d],
                   d.split(':-:')[2], d.split(':-:')[3]]
            parr_full.append(arr)

        pre = dict()
        pre['predicted'] = 0
        pre['annotated'] = 0
        for data in aarr:

            if 'annotated' in pre:
                pre['annotated'] += data[2]
            else:
                pre['annotated'] = data[2]

        # cont only correct prediction
        for data in parr:
            if data[0] in annotated_ents:
                for ano in aarr:
                    if (data[0] == ano[0]) and (data[1] == ano[1]) and (data[3] == ano[3]) and (data[4] == ano[4]):
                        if 'predicted' in pre:
                            pre['predicted'] += data[2]
                        else:
                            pre['predicted'] = data[2]

        pred_count = 0
        correct_pred = 0
        predall_dict = {}
        predall_dict.clear()
        pred_unique_dict = {}
        pred_unique_dict.clear()

        all_prediction_count = 0
        correct_predicition_count = 0
        for data in parr_full:
            all_prediction_count += 1
            if data[0] in annotated_ents:
                for ano in aarr:
                    if (data[0] == ano[0]) and (data[1] == ano[1]) and (data[3] == ano[3]) and (data[4] == ano[4]):
                        correct_predicition_count += 1

        correct_pred = correct_predicition_count

        # All count with only unique count
        pred_count = all_prediction_count

        pre_error = dict()
        pre_error['predicted'] = pred_count
        pre_error['error'] = pred_count - correct_pred

        if pred_count > 0:
            percentage = ((pred_count - correct_pred) / pred_count) * 100
        else:
            percentage = 0

        pre_error['percentage'] = percentage
        irecall = percentage

        annoted_count = pre['annotated']
        correct_pred = pre['predicted']

        pre_precision = {}

        pre_precision['anotated'] = annoted_count
        pre_precision['predicted'] = correct_pred

        if correct_pred > 0:
            percentage = (correct_pred / annoted_count) * 100
        else:
            percentage = 0

        percentage = (correct_pred / annoted_count) * 100
        pre_precision['percentage'] = percentage
        iprecision = percentage
        recall_percentage = (correct_pred / pred_count) * 100
        # orginal F1 score formula
        f1 = 2 * (iprecision * recall_percentage) / (iprecision + recall_percentage)

        ret_data.clear()
        ret_data["precision"] = pre_precision
        ret_data["recall"] = pre_error
        ret_data["efficiency"] = f1

        return ret_data

    def basic_summary(self, anot_dic, pred_dict):
        ret_data = []

        # parse anot_dic to array
        aarr = []
        for d in anot_dic:
            arr = [d.split(':-:')[1], d.split(':-:')[0], anot_dic[d],
                   d.split(':-:')[2], d.split(':-:')[3]]
            aarr.append(arr)

        # create summary of annotated entity count
        ret_anot_dic = {}
        for data in aarr:
            if data[0] in ret_anot_dic:
                ret_anot_dic[data[0]] += 1
            else:
                ret_anot_dic[data[0]] = 1

        dict_test = {}
        for key in anot_dic:
            predicted_count = 0
            ss = key.split(':-:')[1]
            if key in pred_dict:
                if ss in dict_test:
                    dict_test[ss] += 1
                else:
                    dict_test[ss] = 1
        for key in ret_anot_dic:
            ss = key
            # if ss not in str(ret_data):
            anotated_count = ret_anot_dic[ss]
            if ss in dict_test:
                predicted_count = dict_test[ss]
            else:
                predicted_count = 0
            data_dict = {}
            data_dict.clear()
            data_dict["entity"] = ss
            data_dict["annotated"] = anotated_count
            data_dict["predicted"] = predicted_count
            ret_data.append(data_dict)

        return ret_data

    def basic_matrix(self, anot_dic, pred_dict):

        # parsing anot_dic to array
        annotated_ents = []
        aarr = []
        for d in anot_dic:
            arr = [d.split(':-:')[1], d.split(':-:')[0], anot_dic[d],
                   str(d.split(':-:')[2]), str(d.split(':-:')[3])]
            aarr.append(arr)
            annotated_ents.append(d.split(':-:')[1])
        annotated_ents = list(set(annotated_ents))
        # Creating dataframe (for easy quering)
        columns = ['a_ent', 'a_val', 'a_cnt', 'a_start', 'a_end']
        adf = pd.DataFrame(aarr, columns=columns)

        # parsing pred_dict to array
        parr = []
        for d in pred_dict:
            arr = [d.split(':-:')[1], d.split(':-:')[0], pred_dict[d], str(d.split(':-:')[2]),
                   str(d.split(':-:')[3])]
            # arr.append(str(d.split(':-:')[4]))
            parr.append(arr)

        # Creating dataframe (for easy quering)
        columns = ['p_ent', 'p_val', 'p_cnt', 'p_start', 'p_end']
        pdf = pd.DataFrame(parr, columns=columns)

        # create Matrix dataframe
        columns = []
        for key in anot_dic:
            a1 = key.split(':-:')
            columns.append(a1[1])

        columns = list(set(columns))
        columns.append('others')
        index = columns
        df = pd.DataFrame(index=index, columns=columns)
        df.fillna(0, inplace=True)

        # 1 correctly predicted
        df1 = pd.merge(adf, pdf, how='inner', left_on=['a_ent', 'a_val', 'a_start', 'a_end'],
                       right_on=['p_ent', 'p_val', 'p_start', 'p_end'])

        a = pd.merge(pdf, df1, how='outer', indicator=True)
        df2 = a[a['_merge'] == 'left_only']
        df_copy = df2

        del df2['_merge']
        c = pd.concat([df1, df2])  # print(c)
        a = pd.merge(adf, c, how='outer', indicator=True)

        df3 = a[a['_merge'] == 'left_only']

        df_json = df.to_json()

        data_matrix = json.loads(df_json)
        # 1 correctly predicted to matrix
        for index, row in df1.iterrows():
            data_matrix[row['a_ent']][row['p_ent']] += 1

        # 2 annotated entity is predicted as something else to Matrix
        for index, row in df2.iterrows():
            if row['p_ent'] in annotated_ents:
                data_matrix['others'][row['p_ent']] += 1
            else:
                data_matrix['others']['others'] += 1

        # df
        #             StartDate   IRNo    Severity    ORG     others
        # StartDate   0           0        1          0        0
        # IRNo        0           0        1          0        0
        # Severity    0           0        0          0        0
        # ORG         0           0        0          22       0
        # others      0           2        0          0        0
        # ------------> Save to Table 2 in UI.------> Matrix involve Heat Map in UI(color deviation as per the values )

        result_df = data_matrix
        # print(result_df)
        return result_df

    def evaluate(self, anotated_doc, predicted_doc, ithreshold=0.10):

        summary_prediction_set = set([])
        anot_dic = {}
        pred_dict = {}
        all_prediction_dict = {}
        anot_dic.clear()
        anotated_data = []
        ret_data = {}

        for doc in anotated_doc:
            for ent in doc[1]['entities']:
                dat = []
                text = doc[0][ent[0]:ent[1]]
                dat.append(text)
                dat.append(ent[2])
                anotated_data.append(dat)

                str1 = text + ":-:" + ent[2] + ":-:" + str(ent[0]) + ":-:" + str(ent[1])
                if str1 in anot_dic:
                    anot_dic[str1] = 1
                else:
                    anot_dic[str1] = 1

        pred_dict.clear()
        all_prediction_dict.clear()
        try:
            for pdoc in predicted_doc:
                for score in pdoc[1]:
                    if score[0] >= ithreshold:
                        idx = -1
                        for ent in score[1]:
                            idx += 1
                            try:
                                loc_arr = score[2][idx]
                            except Exception as e:
                                print(e, "---------", idx, "-------", pdoc)
                            # if ent[0] in str(anotated_data):
                            # all_prediction_dict
                            str2 = ent[0] + ":-:" + ent[1] + ":-:" + str(loc_arr[0]) + ":-:" + str(loc_arr[1])
                            # str = ent[0] + ":-:" + ent[1]
                            if str2 in all_prediction_dict:
                                all_prediction_dict[str2] += 1
                            else:
                                all_prediction_dict[str2] = 1

                            for anot in anotated_data:
                                if (anot[0] == ent[0]) and (anot[1] == ent[1]):
                                    str1 = ent[0] + ":-:" + ent[1] + ":-:" + str(loc_arr[0]) + ":-:" + str(loc_arr[1])
                                    # str = ent[0] + ":-:" + ent[1]
                                    if str1 in pred_dict:
                                        pred_dict[str1] = 1
                                    else:
                                        pred_dict[str1] = 1

        except Exception as error:
            print(error)

            # end of threshold check

        # pred_dict = {'Amadeus:-:ORG': 22}
        # print(pred_dict)

        ret_data.clear()
        ret_data["annotated"] = anot_dic
        ret_data["predicted"] = pred_dict
        ret_data["full_predict"] = all_prediction_dict
        return ret_data

    def advanced_view(self, anotated_doc, predicted_doc, ithreshold):

        ret_data = {}
        eval_data = self.evaluate(anotated_doc, predicted_doc, ithreshold)
        anot_dic = eval_data['annotated']

        aarr = []
        for d in anot_dic:
            arr = [d.split(':-:')[1], d.split(':-:')[0], anot_dic[d],
                   d.split(':-:')[2], d.split(':-:')[3]]
            aarr.append(arr)

        annotated_ents = []
        anotated_data_arr = []
        for doc in anotated_doc:
            ent_count = 0
            ent_arr = []
            for ent in doc[1]['entities']:
                ent_count += 1
                ent_text = doc[0][ent[0]:ent[1]]
                each_entiry_data = dict()
                each_entiry_data["ent_text"] = ent_text
                each_entiry_data["ent_label"] = ent[2]
                each_entiry_data["ent_start"] = ent[0]
                each_entiry_data["ent_end"] = ent[1]
                ent_arr.append(each_entiry_data)

                annotated_ents.append(ent[2])

            anotated_data = dict()
            anotated_data["text"] = doc[0]
            anotated_data["entities"] = ent_arr
            anotated_data["ent_count"] = ent_count
            anotated_data.append(anotated_data)

        annotated_ents = list(set(annotated_ents))

        all_prediction_dict = {}
        all_prediction_dict.clear()

        predicted_data_arr = []
        try:
            for pdoc in predicted_doc:
                each_predicted_score_arr = []
                for score in pdoc[1]:
                    if score[0] >= ithreshold:
                        idx = -1
                        ent_arr_final = []
                        for ent in score[1]:
                            idx += 1
                            try:
                                loc_arr = score[2][idx]
                            except Exception as e:
                                print(e, "---------", idx, "-------", pdoc)

                            each_predict_data = {}

                            if ent[1] in annotated_ents:

                                each_predict_data["is_annotated"] = True
                                each_predict_data["is_predicted"] = False
                                for ano in aarr:
                                    if (ent[0] == ano[1]) and (ent[1] == ano[0]) and (loc_arr[0] == int(ano[3])) and (
                                            loc_arr[1] == int(ano[4])):
                                        each_predict_data["is_predicted"] = True

                            else:
                                # not a anotated entity
                                each_predict_data["is_annotated"] = False

                            # Each_predictData["score"] = score[0]
                            each_predict_data["predict_text"] = ent[0]
                            each_predict_data["predict_label"] = ent[1]
                            each_predict_data["predict_start"] = loc_arr[0]
                            each_predict_data["predict_end"] = loc_arr[1]
                            ent_arr_final.append(each_predict_data)

                        each_predicted_score = {'score': round(score[0], 2), 'entities': ent_arr_final}
                        each_predicted_score_arr.append(each_predicted_score)

                prediction_final_dict = {'text': pdoc[0], 'scores': each_predicted_score_arr}
                predicted_data_arr.append(prediction_final_dict)

        except Exception as error:
            print(error)

        if predicted_data_arr is not None:
            for a in anotated_data_arr:
                for p in predicted_data_arr:
                    if a['text'] == p['text']:
                        # print("-----------=========>",a['Text'])
                        a['prediction'] = p['scores']
        ret_data['annotation'] = anotated_data_arr

        return ret_data
