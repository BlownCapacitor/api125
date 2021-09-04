from flask import Flask, jsonify, request
from classifier import getPrediction2

app = Flask(__name__)

@app.route("/predict-digit", methods = ['POST'])

def predictData():
    image = request.files.get("digit")
    prediction = getPrediction2(image)
    return jsonify({
        'prediction': prediction,
      }), 200

if __name__ == '__main__':
    app.run(debug = True)

    