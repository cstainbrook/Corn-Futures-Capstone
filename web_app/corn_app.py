from flask import Flask, render_template
import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/research')
def research():
    return render_template('research.html')

@app.route('/pastpredictions')
def pastpreds():
    return render_template('pastpredictions.html')

@app.route('/currentpredictions')
def currentpreds():
    return render_template('currentpredictions.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8102, debug=False, threaded=True)
