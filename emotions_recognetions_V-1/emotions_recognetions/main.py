
from flask import Flask
from importlib import import_module

def create_app(env):
    app = Flask(__name__)

    module = import_module('appConf')
    EnvConfig = getattr(module, env)
    app.config.from_object(EnvConfig)

    api_version = import_module('app')
    app.register_blueprint(getattr(api_version, "api_bp"))

    return app

if __name__ == '__main__':
    app = create_app('Production')
    app.run(host='0.0.0.0', port=9191, threaded=True)