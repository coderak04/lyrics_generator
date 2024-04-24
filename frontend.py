from flask import Flask, render_template, request
from main import *

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    token_maker, max_sequence_length,model = load_data_and_model()
    if request.method == 'POST':
        starting_text = request.form['seed_text']
        generated_lyrics = complete_this_song(model, token_maker, max_sequence_length, starting_text,50)
        print(generated_lyrics)
        return render_template('index.html', generated_lyrics=generated_lyrics, show_result=True)
    return render_template('index.html', show_result=False)

if __name__ == '__main__':
    app.run(debug=True)