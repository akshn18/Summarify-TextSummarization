from flask import Flask, render_template, request
from spacy_summarization import text_summarizer
import time
import spacy

nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

# Reading Time
def readingTime(mytext):
    total_words = len([token.text for token in nlp(mytext)])
    estimatedTime = total_words / 200.0
    return estimatedTime

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        final_reading_time = readingTime(rawtext)
        final_summary = text_summarizer(rawtext)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end - start
    return render_template('index.html', ctext=rawtext, final_summary=final_summary, final_time=final_time, final_reading_time=final_reading_time, summary_reading_time=summary_reading_time)

@app.route('/about')
def about():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
