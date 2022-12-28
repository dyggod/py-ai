from flask import Flask, redirect, url_for, render_template
# from gevent import pywsgi
from flask_jwt_extended import get_jwt_identity, jwt_required, JWTManager

from some_route import addRoute
from routes.login import login
from routes.upload import uploadRoute

app = Flask(
    __name__,
    static_folder='./dist',  # 静态文件目录
    template_folder='./dist',  # 模板文件目录
    static_url_path='')  # 静态文件url前缀，表明静态文件的url是从static_folder的值开始的
port = 3009

# 定义文件上传的位置
app.config['UPLOAD_FOLDER'] = 'uploads/'

# 定义jwt
app.config['JWT_SECRET_KEY'] = 'ai-py-flask-jwt-secret'
jwt = JWTManager(app)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/hello')
def hello():
    return 'Hello!'


@app.route('/hello/<name>')
def hello_name(name):
    return 'Hello %s!' % name


@app.route('/blog/<int:postID>')
def show_blog(postID):
    return 'Blog Number %d' % postID


@app.route('/rev/<float:revNo>')
def revision(revNo):
    return 'Revision Number %f' % revNo


@app.route('/flask')
def hello_flask():
    return 'Hello Flask'


@app.route('/python/')
def hello_python():
    return 'Hello Python'


@app.route('/user_admin/')
def hello_user_admin():
    return 'Hello user_admin'


@app.route('/user_guest/')
def hello_user_guest():
    return 'Hello user_guest'


@app.route('/user/<name>')
def hello_user(name):
    if name == 'admin':
        return redirect(url_for('hello_user_admin'))
    else:
        return redirect(url_for('hello_user_guest'))


addRoute(app)

login(app)

uploadRoute(app)

# WSGI server
# server = pywsgi.WSGIServer(('0.0.0.0', 3009), app)
# server.serve_forever()

if __name__ == '__main__':
    app.run('0.0.0.0', port)
