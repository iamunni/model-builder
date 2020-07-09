import json
from datetime import datetime
from sqlalchemy import create_engine
from settings import configur

db_url = 'postgresql+psycopg2://%s:%s@%s/%s' %(configur.get('db', 'user'), configur.get('db', 'password'),
                                               configur.get('db', 'host'), configur.get('db', 'name'))
engine = create_engine(db_url, echo=True)
# conn = engine.connect()


class MailbotDB(object):
    def __init__(self):
        self._db_connection = engine.connect()

    def query(self, query, params=None):
        if params:
            return self._db_connection.execute(query, params)
        return self._db_connection.execute(query)

    def __del__(self):
        self._db_connection.close()


# Function to perform the DB Save and Update Function Based on Model Name
def db_save(db_conn, modelname, usecaseid, beamscore, evalscore, outputdir, model_summary, model_matrix,
            model_efficiency, model_accuracy, model_error, model_threshold):
    res = db_conn.query("select id, version from usecases_modelconfig where model_name='%s'" % modelname)
    model_conf = res.fetchone()
    if model_conf:
        db_conn.query("""UPDATE public.usecases_modelconfig SET 
                        model_name=%s, description=%s, threshold=%s, precision_percentage=%s, efficiency=%s, 
                        recall_percentage=%s, version=%s, status=%s, updated=%s, 
                        use_case_id=%s, model_path=%s, beam_score=%s, evaluation_score=%s, 
                        model_matrix=%s, model_summary=%s WHERE id=%s""",
                     (modelname, 'Using the Old Parameters itself', model_threshold, model_accuracy,
                      model_efficiency, model_error, model_conf[1] + 0.1, True,
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S'), usecaseid, get_path(outputdir),
                      str(beamscore), str(evalscore),  json.dumps(model_matrix),
                      json.dumps(model_summary), model_conf[0]))
    else:
        db_conn.query("""INSERT INTO public.usecases_modelconfig(
                        model_name, description, threshold, precision_percentage, efficiency, 
                        recall_percentage, version, status, created, updated,
                        use_case_id, model_path, beam_score, 
                        evaluation_score, model_matrix, model_summary)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                     (modelname, 'Using the Old Parameters itself', model_threshold, model_accuracy,
                      model_efficiency, model_error, 1.0, True, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S'), usecaseid, get_path(outputdir),
                      str(beamscore), str(evalscore), json.dumps(model_matrix),
                      json.dumps(model_summary)))

    print(modelname, "-- Model Saved")


def get_path(full_path):
    return str(full_path).split('modelfiles')[1]
