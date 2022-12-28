def addRoute(app):
    @app.route('/other_route/')
    def otherOne():
        return 'other route'

    @app.route('/other_route/<name>')
    def otherTwo(name):
        return 'other route %s!' % name
