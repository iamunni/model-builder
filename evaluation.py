import json
from spacy.scorer import Scorer
from spacy.gold import GoldParse

from collections import defaultdict
import pandas as pd


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
            return scorer.scores
        except Exception as e:
            print(f"Error due to {e}")

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
        evaluation_score = self.score(nlp, training_data, labels)
        beam_score = self.beam_score(nlp, training_data)
        ret_data = {'evaluation_score': evaluation_score, 'beam_score': beam_score}
        return ret_data

    def basic_view(self, anotated_doc, predicted_doc, ithreshold):
        """
                    Summary of whole Validation Emails
        """

        eval_data = self.evaluate(anotated_doc, predicted_doc, ithreshold)
        anot_dic = eval_data['annotated']
        pred_dict = eval_data['predicted']
        entities_list = eval_data['entities']

        matrix_cal = self.basic_matrix_new(anot_dic, predicted_doc, entities_list, ithreshold)
        matrix = matrix_cal['matrix']
        pre = self.calculate_pre(anot_dic, matrix, ithreshold)
        summary = self.basic_summary(anot_dic, pred_dict, matrix)

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

    def calculate_pre(self, anot_dic, matrix, ithreshold=0.10):
        ret_data = {}

        annotated_ents = []
        aarr = []
        for d in anot_dic:
            arr = [d.split(':-:')[1], d.split(':-:')[0], anot_dic[d],
                   d.split(':-:')[2], d.split(':-:')[3]]
            aarr.append(arr)
            annotated_ents.append(d.split(':-:')[1])

        annotated_ents = list(set(annotated_ents))


        pre = dict()
        pre['predicted'] = 0
        pre['annotated'] = 0
        for data in aarr:

            if 'annotated' in pre:
                pre['annotated'] += data[2]
            else:
                pre['annotated'] = data[2]

        prediction_counter = 0
        for m, k in matrix.items():
            if m != "others":
                print(m, "------", k[m])
                prediction_counter += int(k[m])
        # print(prediction_counter)
        pre['predicted'] = prediction_counter

        total_counter = 0
        for k, v in matrix.items():
            for i, j in v.items():
                print(i, "----", j)
                total_counter += j
        print(total_counter,"----- the Total Count----")
        pre_error = dict()
        pre_error['predicted'] = total_counter
        pre_error['error'] = total_counter - prediction_counter

        if total_counter > 0:
            percentage = ((total_counter - prediction_counter) / total_counter) * 100
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

        pre_precision['percentage'] = percentage
        iprecision = percentage
        recall_percentage = (correct_pred / total_counter) * 100
        # orginal F1 score formula
        f1 = 2 * (iprecision * recall_percentage) / (iprecision + recall_percentage)

        ret_data.clear()
        ret_data["precision"] = pre_precision
        ret_data["recall"] = pre_error
        ret_data["efficiency"] = f1

        return ret_data

    def basic_summary(self, anot_dic, pred_dict, matrix):
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
                ret_anot_dic[data[0]] += data[2]
            else:
                ret_anot_dic[data[0]] = data[2]


        for key in ret_anot_dic:
            ss = key
            anotated_count = ret_anot_dic[ss]
            predicted_count = matrix[ss][ss]
            data_dict = {}
            data_dict.clear()
            data_dict["entity"] = ss
            data_dict["annotated"] = anotated_count
            data_dict["predicted"] = predicted_count
            data_dict["not_recognized"] = anotated_count - predicted_count

            incorrectly_predicted = {}
            incorrectly_predicted['others'] = matrix['others'][ss]

            for p, q in matrix.items():
                if (p != ss) and (p != "others"):
                    print(q[ss],"-----Value")
                    incorrectly_predicted[p] = 3#q[ss]


            data_dict['incorrect'] = incorrectly_predicted

            ret_data.append(data_dict)

        return ret_data

    # def basic_matrix(self, anot_dic, pred_dict):
    #
    #     # parsing anot_dic to array
    #     annotated_ents = []
    #     aarr = []
    #     for d in anot_dic:
    #         arr = [d.split(':-:')[1], d.split(':-:')[0], anot_dic[d],
    #                str(d.split(':-:')[2]), str(d.split(':-:')[3])]
    #         aarr.append(arr)
    #         annotated_ents.append(d.split(':-:')[1])
    #     annotated_ents = list(set(annotated_ents))
    #     # Creating dataframe (for easy quering)
    #     columns = ['a_ent', 'a_val', 'a_cnt', 'a_start', 'a_end']
    #     # Creation of Annotate DataFrame
    #     adf = pd.DataFrame(aarr, columns=columns)
    #
    #     # parsing pred_dict to array
    #     parr = []
    #     for d in pred_dict:
    #         arr = [d.split(':-:')[1], d.split(':-:')[0], pred_dict[d], str(d.split(':-:')[2]),
    #                str(d.split(':-:')[3])]
    #         # arr.append(str(d.split(':-:')[4]))
    #         parr.append(arr)
    #
    #     # Creating dataframe (for easy quering)
    #     columns = ['p_ent', 'p_val', 'p_cnt', 'p_start', 'p_end']
    #     # Creation of Prediction
    #     pdf = pd.DataFrame(parr, columns=columns)
    #
    #     # create Matrix dataframe
    #     columns = []
    #     for key in anot_dic:
    #         a1 = key.split(':-:')
    #         columns.append(a1[1])
    #
    #     columns = list(set(columns))
    #     columns.append('others')
    #     index = columns
    #     df = pd.DataFrame(index=index, columns=columns)
    #     df.fillna(0, inplace=True)
    #
    #     # 1 correctly predicted
    #     df1 = pd.merge(adf, pdf, how='inner', left_on=['a_ent', 'a_val', 'a_start', 'a_end'],
    #                    right_on=['p_ent', 'p_val', 'p_start', 'p_end'])
    #
    #     a = pd.merge(pdf, df1, how='outer', indicator=True)
    #     df2 = a[a['_merge'] == 'left_only']
    #     df_copy = df2
    #
    #     del df2['_merge']
    #     c = pd.concat([df1, df2])  # print(c)
    #     a = pd.merge(adf, c, how='outer', indicator=True)
    #
    #     df3 = a[a['_merge'] == 'left_only']
    #
    #     df_json = df.to_json()
    #
    #     data_matrix = json.loads(df_json)
    #     # 1 correctly predicted to matrix
    #     for index, row in df1.iterrows():
    #         data_matrix[row['a_ent']][row['p_ent']] += 1
    #
    #     # 2 annotated entity is predicted as something else to Matrix
    #     # for index, row in df2.iterrows():
    #     #     if row['p_ent'] in annotated_ents:
    #     #         data_matrix['others'][row['p_ent']] += 1
    #     #         for a in adf:
    #     #             if a['val'] == row['p_val']:
    #     #                 data_matrix['others'][row['p_ent']] += 1
    #     #             else:
    #     #                 data_matrix['others'][row['p_ent']] += 1
    #     #     else:
    #     #         data_matrix['others']['others'] += 1
    #
    #     # print(adf.info())
    #     for index, row in df2.iterrows():
    #         if row['p_ent'] in annotated_ents:
    #             if row['p_ent']=='IRNo':
    #                 print("---Test---",row)
    #             flag = False
    #             for idx, adf_row in adf.iterrows():
    #                 if adf_row['a_val'] == row['p_val']:
    #                     data_matrix[a['a_ent']][row['p_ent']] += 1 #1
    #                     flag = True
    #             if flag is False:
    #                 data_matrix['others'][row['p_ent']] += 1 #3
    #         else:
    #             data_matrix['others']['others'] += 1 #3
    #
    #     # df
    #     #             StartDate   IRNo    Severity    ORG     others
    #     # StartDate   0           0        1          0        0
    #     # IRNo        0           0        1          0        0
    #     # Severity    0           0        0          0        0
    #     # ORG         0           0        0          22       0
    #     # others      0           2        0          0        0
    #     # ------------> Save to Table 2 in UI.------> Matrix involve Heat Map in UI(color deviation as per the values )
    #
    #     result_df = data_matrix
    #     # print(result_df)
    #     return result_df

    def basic_matrix_new(self, anot_dic, predicted_doc, annotated_ents, ithreshold):

        ret_data = {}
        matrix_dic = {}

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
        df_json = df.to_json()
        # Data Matrix is created here....
        data_matrix = json.loads(df_json)


        ################-------- CODE CHANGE-------- NEED TO RECREATE THE MATRIX LOGIC HERE-----
        aarr = []
        for d in anot_dic:
            arr = [d.split(':-:')[1], d.split(':-:')[0], anot_dic[d],
                   d.split(':-:')[2], d.split(':-:')[3]]
            aarr.append(arr)

        all_prediction_dict = {}
        all_prediction_dict.clear()

        try:
            val_count = 0
            for pdoc in predicted_doc:
                val_count += 1
                for score in pdoc[1]:
                    if score[0] >= ithreshold:
                        idx = -1
                        for ent in score[1]:
                            idx += 1
                            try:
                                loc_arr = score[2][idx]
                            except Exception as e:
                                print(e, "---------", idx, "-------", pdoc)

                            each_predict_data = {}

                            if ent[1] in annotated_ents:
                                flag = False
                                for ano in aarr:
                                    if (ent[0] == ano[1]) and (ent[1] == ano[0]) and (loc_arr[0] == int(ano[3])) and (
                                            loc_arr[1] == int(ano[4])):
                                        str1 = ent[1] + ":-:" + ano[0] + ":-:" + str(val_count) + ":-:" + ent[0]
                                        if str1 in matrix_dic:
                                            matrix_dic[str1] += 1
                                        else:
                                            matrix_dic[str1] = 1
                                        flag = True
                                if flag is False:
                                    for ano in aarr:
                                        if (ent[0] == ano[1]) and (loc_arr[0] == int(ano[3])) and (
                                                loc_arr[1] == int(ano[4])):
                                            str1 = ent[1] + ":-:" + ano[0] + ":-:" + str(val_count) + ":-:" + ent[0]
                                            if str1 in matrix_dic:
                                                matrix_dic[str1] += 1
                                            else:
                                                matrix_dic[str1] = 1
                                            flag = True
                                    if flag is False:
                                        str1 = "others" + ":-:" + ent[1] + ":-:" + str(val_count) + ":-:" + ent[0]

                                        if str1 in matrix_dic:
                                            matrix_dic[str1] += 1
                                        else:
                                            matrix_dic[str1] = 1
                            else:
                                str1 = "others" + ":-:" + "others" + ":-:" + str(val_count) + ":-:" + ent[0]
                                if str1 in matrix_dic:
                                    matrix_dic[str1] += 1
                                else:
                                    matrix_dic[str1] = 1
        except Exception as error:
            print(error)

        for d in matrix_dic:
            ent_1 = d.split(':-:')[1]
            ent_2 = d.split(':-:')[0]
            data_matrix[ent_2][ent_1] += 1

        ret_data['matrix'] = data_matrix
        return ret_data

    def evaluate(self, anotated_doc, predicted_doc, ithreshold=0.10):

        anot_dic = {}
        pred_dict = {}
        all_prediction_dict = {}
        anot_dic.clear()
        anotated_data = []
        ret_data = {}
        annotated_ents = []

        for doc in anotated_doc:
            for ent in doc[1]['entities']:
                dat = []
                text = doc[0][ent[0]:ent[1]]
                dat.append(text)
                dat.append(ent[2])
                anotated_data.append(dat)
                annotated_ents.append(ent[2])

                str1 = text + ":-:" + ent[2] + ":-:" + str(ent[0]) + ":-:" + str(ent[1])
                if str1 in anot_dic:
                    anot_dic[str1] += 1
                else:
                    anot_dic[str1] = 1
        annotated_ents = list(set(annotated_ents))
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
                            str2 = ent[0] + ":-:" + ent[1] + ":-:" + str(loc_arr[0]) + ":-:" + str(loc_arr[1])
                            if str2 in all_prediction_dict:
                                all_prediction_dict[str2] += 1
                            else:
                                all_prediction_dict[str2] = 1

                            for anot in anotated_data:
                                if (anot[0] == ent[0]) and (anot[1] == ent[1]):
                                    str1 = ent[0] + ":-:" + ent[1] + ":-:" + str(loc_arr[0]) + ":-:" + str(loc_arr[1])
                                    if str1 in pred_dict:
                                        pred_dict[str1] += 1
                                    else:
                                        pred_dict[str1] = 1

        except Exception as error:
            print(error)

        ret_data.clear()
        ret_data["annotated"] = anot_dic
        ret_data["predicted"] = pred_dict
        ret_data["full_predict"] = all_prediction_dict
        ret_data["entities"] = annotated_ents
        return ret_data

    def advanced_view(self, anotated_doc, predicted_doc, ithreshold):

        ret_data = {}
        eval_data = self.evaluate(anotated_doc, predicted_doc, ithreshold)
        anot_dic = eval_data['annotated']

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
        df_json = df.to_json()
        # Data Matrix is created here....
        data_matrix = json.loads(df_json)

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
            anotated_data_arr.append(anotated_data)

        annotated_ents = list(set(annotated_ents))

        all_prediction_dict = {}
        all_prediction_dict.clear()
        matrix_dic = {}
        predicted_data_arr = []
        try:
            val_count = 0
            for pdoc in predicted_doc:
                val_count += 1
                each_predicted_score_arr = []
                for score in pdoc[1]:
                    if score[0] >= ithreshold:
                        idx = -1
                        ent_arr_final = []
                        for ent in score[1]:
                            anot_label = ''
                            predict_label = ''
                            idx += 1
                            try:
                                loc_arr = score[2][idx]
                            except Exception as e:
                                print(e, "---------", idx, "-------", pdoc)

                            each_predict_data = {}

                            if ent[1] in annotated_ents:
                                each_predict_data["is_annotated"] = True
                                each_predict_data["is_predicted"] = False
                                flag = False
                                for ano in aarr:
                                    if (ent[0] == ano[1]) and (ent[1] == ano[0]) and (loc_arr[0] == int(ano[3])) and (
                                            loc_arr[1] == int(ano[4])):
                                        each_predict_data["is_predicted"] = True


                                        str1 = ent[1] + ":-:" + ano[0] + ":-:" + str(val_count) + ":-:" + ent[0]
                                        if str1 in matrix_dic:
                                            matrix_dic[str1] += 1
                                        else:
                                            matrix_dic[str1] = 1

                                        anot_label = ano[0]
                                        predict_label = ent[1]
                                        flag = True
                                if flag is False:
                                    for ano in aarr:
                                        if (ent[0] == ano[1]) and (loc_arr[0] == int(ano[3])) and (loc_arr[1] == int(ano[4])):

                                            str1 = ent[1] + ":-:" + ano[0] + ":-:" + str(val_count) + ":-:" + ent[0]
                                            if str1 in matrix_dic:
                                                matrix_dic[str1] += 1
                                            else:
                                                matrix_dic[str1] = 1

                                            anot_label = ano[0]
                                            predict_label = ent[1]
                                            flag = True
                                    if flag is False:
                                        str1 = "others" + ":-:" + ent[1] + ":-:" + str(val_count) + ":-:" + ent[0]

                                        if str1 in matrix_dic:
                                            matrix_dic[str1] += 1
                                        else:
                                            matrix_dic[str1] = 1

                                        anot_label = 'others'
                                        predict_label = ent[1]
                            else:
                                str1 = "others" + ":-:" + "others" + ":-:" + str(val_count) + ":-:" + ent[0]
                                if str1 in matrix_dic:
                                    matrix_dic[str1] += 1
                                else:
                                    matrix_dic[str1] = 1

                                anot_label = 'others'
                                predict_label = 'others'
                                each_predict_data["is_annotated"] = False
                            if each_predict_data["is_predicted"] == False and ent[1] == 'IRNo':
                                print("-------IR Test----")
                            each_predict_data["predict_text"] = ent[0]
                            each_predict_data["predict_label"] = predict_label
                            each_predict_data["annotated_label"] = anot_label
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
                        a['prediction'] = p['scores']
        ret_data['annotation'] = anotated_data_arr

        for d in matrix_dic:
            ent_1 = d.split(':-:')[1]
            ent_2 = d.split(':-:')[0]
            data_matrix[ent_2][ent_1] += 1
        ret_data['matrix'] = data_matrix
        return ret_data


    def compare_view(self, anotated_doc, models_config):

        modelnames_arr =[]

        precision_arr = []
        recall_arr = []
        efficiency_arr = []


        for model_config in models_config:

            modelnames_arr.append(model_config.model_name)

            ithreshold = model_config.threshold
            predicted_doc = model_config.beam_score

            eval_data = self.evaluate(anotated_doc, predicted_doc, ithreshold)
            anot_dic = eval_data['annotated']
            # pred_dict = eval_data['predicted']
            entities_list = eval_data['entities']

            matrix_cal = self.basic_matrix_new(anot_dic, predicted_doc, entities_list, ithreshold)
            matrix = matrix_cal['matrix']
            pre = self.calculate_pre(anot_dic, matrix, ithreshold)
            # summary = self.basic_summary(anot_dic, pred_dict, matrix)

            precision = pre['precision']['percentage']
            recall = pre['recall']['percentage']
            efficiency = pre['efficiency']

            precision_arr.append(precision)
            recall_arr.append(recall)
            efficiency_arr.append(efficiency)


        compare_data = dict()
        compare_data["precision"] = precision_arr
        compare_data["recall"] = recall_arr
        compare_data["efficiency"] = efficiency_arr
        compare_data["models"] = modelnames_arr

        # basic_data["summary"] = summary
        # basic_data["matrix"] = matrix

        return compare_data

