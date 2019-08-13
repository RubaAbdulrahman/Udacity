import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [
    {
      "uid": "a089bd",
      "line": {},
      "name": "Col2",
      "type": "bar",
      "x":genre_counts,
      "y":genre_names,
      "marker": {
        "line": {
          "width": 1
        },
        "color": "rgb(70, 162, 187)"
      },
      "error_x": {},
      "error_y": {},
      "textfont": {},
      "orientation": "h"
    }
  ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Genre"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        },
        
        # GRAPH 2 - category graph
        #Reference: https://plot.ly/~C_Sevigny/4#plot
        {
  "data": [
    {
      "uid": "a089bd",
      "line": {},
      "name": "Col2",
      "type": "bar",
      "x": category_boolean,
      "y":category_names,
      "marker": {
        "line": {
          "width": 1
        },
        "color": "rgb(70, 162, 187)"
      },
      "error_x": {},
      "error_y": {},
      "textfont": {},
      "orientation": "h"
    }
  ],
  "layout": {
    "font": {
      "size": 12,
      "color": "rgb(33, 33, 33)",
      "family": "Raleway, sans-serif"
    },
    "smith": False,
    "title": "<br>Distribution of Message Categories",
    "width": 800,
    "xaxis": {
      "type": "linear",
      "dtick": 20000,
      "range": [
        0,
        103741.05263157895
      ],
      "tick0": 0,
      "ticks": "",
      "title": "<br><i>Data:Message data for disaster response</i>",
      "anchor": "y",
      "domain": [
        0,
        1
      ],
      "mirror": False,
      "nticks": 0,
      "ticklen": 5,
      "autotick": True,
      "position": 0,
      "showgrid": True,
      "showline": False,
      "tickfont": {
        "size": 0,
        "color": "",
        "family": ""
      },
      "zeroline": False,
      "autorange": True,
      "gridcolor": "rgb(255, 255, 255)",
      "gridwidth": 1,
      "linecolor": "#000",
      "linewidth": 1,
      "rangemode": "normal",
      "tickangle": 40,
      "tickcolor": "#000",
      "tickwidth": 1,
      "titlefont": {
        "size": 0,
        "color": "",
        "family": ""
      },
      "overlaying": False,
      "showexponent": "all",
      "zerolinecolor": "#000",
      "zerolinewidth": 1,
      "exponentformat": "none",
      "showticklabels": True
    },
    "yaxis": {
      "type": "category",
      "dtick": 1,
      "range": [
        -0.5,
        12.5
      ],
      "tick0": 0,
      "ticks": "",
      "title": "",
      "anchor": "x",
      "domain": [
        0,
        1
      ],
      "mirror": False,
      "nticks": 0,
      "ticklen": 5,
      "autotick": True,
      "position": 0,
      "showgrid": False,
      "showline": False,
      "tickfont": {
        "size": 0,
        "color": "",
        "family": ""
      },
      "zeroline": False,
      "autorange": True,
      "gridcolor": "#ddd",
      "gridwidth": 1,
      "linecolor": "#000",
      "linewidth": 1,
      "rangemode": "normal",
      "tickangle": "auto",
      "tickcolor": "#000",
      "tickwidth": 1,
      "titlefont": {
        "size": 0,
        "color": "",
        "family": ""
      },
      "overlaying": False,
      "showexponent": "all",
      "zerolinecolor": "#000",
      "zerolinewidth": 1,
      "exponentformat": "e",
      "showticklabels": True
    },
    "bargap": 0.36,
    "boxgap": 0.3,
    "height": 600,
    "legend": {
      "x": 1.02,
      "y": 1,
      "font": {
        "size": 0,
        "color": "",
        "family": ""
      },
      "bgcolor": "#fff",
      "xanchor": "left",
      "yanchor": "top",
      "traceorder": "normal",
      "bordercolor": "#000",
      "borderwidth": 1
    },
    "margin": {
      "b": 80,
      "l": 320,
      "r": 80,
      "t": 80,
      "pad": 0,
      "autoexpand": True
    },
    "barmode": "stack",
    "boxmode": "overlay",
    "autosize": False,
    "dragmode": "zoom",
    "hovermode": "x",
    "titlefont": {
      "size": 0,
      "color": "",
      "family": ""
    },
    "separators": ".,",
    "showlegend": False,
    "bargroupgap": 0.02,
    "boxgroupgap": 0.3,
    "hidesources": False,
    "plot_bgcolor": "rgba(102, 102, 102, 0.18)",
    "paper_bgcolor": "rgb(255, 255, 255)"
  },
  "frames": []
}
    ]
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
