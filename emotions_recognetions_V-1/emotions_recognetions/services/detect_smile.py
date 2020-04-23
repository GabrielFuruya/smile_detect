from flask import request
from flask_restful import Resource
from routines.routines import Classifie

class DetectSmile(Resource):
    def __init__(self):
        pass

    def get(self):
        return {"message": "GET is not supported"}, 401
    
    def post(self):
        # try:
        json_data = request.get_json(force=True)
        # TODO: Routine to predict img
        base64 = json_data['base_64']
        file_name = json_data['filename']
        smile = Classifie().run(base64,file_name)
        return {"message": "Success in process", "data":smile}, 200

        # except Exception as e:
        #     print(e)
        #     return {"message": f"Error to process, {str(e)}", "data":None}, 404

