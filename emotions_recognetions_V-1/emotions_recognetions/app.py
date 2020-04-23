from flask import Blueprint
from flask_restful import Api
from services.detect_smile import DetectSmile

api_bp = Blueprint('api', __name__)
api = Api(api_bp)
api.add_resource(DetectSmile, '/detect_smile')