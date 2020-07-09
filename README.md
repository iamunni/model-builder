# README #


### How do I get set up? ###

* Create virtual env
* pip install -r requirements.txt
* Create a folder named 'modelfiles/usecases'
* Create a folder named 'modelfiles/classifiers'
* python classifier.py <application_id>
* python entity_trainer_1.py <usecase_id>
* python entity_trainer_2.py <usecase_id>

### For testing add '--source csv'
* Eg: python entity_trainer_1.py <usecase_id> --source csv
* Eg: python classifier.py <application_id> --source csv
