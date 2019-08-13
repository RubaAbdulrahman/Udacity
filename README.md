# Udacity
## Data Science Nanodegree
### Project: Disaster Response Pipeline
---
#### Project Overview
 In this project, I'll apply data engineering skills to analyze disaster data from <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> to build a model for an API that classifies disaster messages.

#### How to run the Python scripts & Wep app
**From the project directory** run the following command:
```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
After the data cleaning process, run this command:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

So we have cleaned the data and trained our model. to run the Wep App

**Go the app directory** and run the following command:

<a id='com'></a>

```bat
python run.py
```

This will start the web app that will pridict and classify the message.

#### Software Requirements

This project uses **Python 3.6.6**.
