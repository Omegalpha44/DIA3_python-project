import io
import flask
import matplotlib.pyplot as plt
import numpy as np

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return "<h1>Bonjour</h1><p>Ceci est une page de test</p>"

@app.route('/api/v1/test', methods=['GET'])
def api_all():
    return "<h1>Bonjour</h1><p>Ceci est la seconde page de test</p>"

@app.route('/api/v1/plot', methods=['GET'])
def plot():
    x = np.linspace(-10, 10, 100)
    y = x**2
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('y = x^2')
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return flask.send_file(img, mimetype='image/png')


app.run()