from flask import request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
import json

# 写一个登录接口


def login(app):
    @app.route('/login/', methods=['GET', 'POST'])
    def login():
        admin_dict = {'name': 'admin', 'age': 18}
        not_admin_str = 'user is not admin'
        error_response = f'login error: {not_admin_str}'
        if request.method == 'POST':
            content_type = request.headers['Content-Type']
            print('*************************')
            print(content_type)
            print('*************************')
            data = dict()
            if content_type == 'application/json':
                data = json.loads(request.get_data(as_text=True))
            elif content_type == 'application/x-www-form-urlencoded':
                data = request.form
            else:
                data = request.form
            if data['username'] == 'admin':
                access_token = create_access_token(identity='admin')
                admin_dict['access_token'] = access_token
                return jsonify(admin_dict)
            else:
                return error_response
        else:
            user = request.args.get('username')
            if user == 'admin':
                access_token = create_access_token(identity='admin')
                admin_dict['access_token'] = access_token
                return jsonify(admin_dict)
            else:
                return error_response

    @app.route('/get_user/', methods=['GET'])
    @jwt_required()
    def get_user():
        current_user = get_jwt_identity()
        return jsonify(logged_in_as=current_user), 200
