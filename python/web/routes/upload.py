from flask import request, jsonify
from werkzeug.utils import secure_filename
import os

# 写一个上传文件的接口


def uploadRoute(app):
    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            f = request.files['file']
            # 如果app.config['UPLOAD_FOLDER']不存在，则创建该目录
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            f.save(
                os.path.join(app.config['UPLOAD_FOLDER'],
                             secure_filename(f.filename)))
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'fail', 'msg': 'not post'})
