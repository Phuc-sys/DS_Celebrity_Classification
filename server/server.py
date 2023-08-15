from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/classify_img', methods=['GET', 'POST'])
def classify_img():
    img_data = request.form['img_data']

    response = jsonify(util.classify_img(img_data))
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == '__main__':
    print('Starting Python Flask Server')
    util.load_saved_artifacts()
    app.run(port=5000)