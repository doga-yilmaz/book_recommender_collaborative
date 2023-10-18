from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the recommendation model
with open('artifacts/model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('artifacts/book_names.pkl', 'rb') as file:
    book_names = pickle.load(file)

with open('artifacts/book_pivot.pkl', 'rb') as file:
    book_pivot = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['book_name']
    recommendations = recommend_book(user_input)
    return render_template('index.html', recommendations=recommendations)

def recommend_book(book_name):
    recommendations = []
    # Your recommendation logic here
    book_id = np.where(book_names == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

    for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                recommendations.append(j)

    return recommendations

if __name__ == '__main__':
    app.run(debug=True)