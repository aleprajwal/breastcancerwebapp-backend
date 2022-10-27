import os
from flask import Flask
from flask_restful import Api
from flask_cors import CORS

from classifier import BreastCancerCassifier

app = Flask(__name__)
api = Api(app)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

api.add_resource(BreastCancerCassifier, '/classifier')


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, port=port)
