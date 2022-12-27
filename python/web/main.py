from flask import Flask
# from gevent import pywsgi

app = Flask(__name__)
port = 3009

@app.route('/')
def hello_world():
    return 'Hello World!'

# WSGI server
# server = pywsgi.WSGIServer(('0.0.0.0', 3009), app)
# server.serve_forever()

if __name__ == '__main__':
    app.run('0.0.0.0', port)
