import json
import os
from datetime import datetime
import warnings

class Logger():
    def __init__(self, log_path, settings):
        log_file = f"/log_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.json"
        self.log_file = log_path + log_file
        self._set_up_log(settings)

    def _write_json(self, data): 
        with open(self.log_file,'w') as f: 
            json.dump(data, f, indent=4) 
        
    def log(self, results):
        with open(self.log_file) as json_file: 
            data = json.load(json_file) 
            temp = data['output'] 
            temp.append(results) 
        self._write_json(data)  

    def _set_up_log(self, opt):
        try:
            args = vars(opt)
        except:
            warnings.warn("No __dict__ found in Object")
            args = []

        empty_log = {
            "meta_data" : {
                "file": os.path.basename(__file__),
                "datetime": str(datetime.now()),
                "args": args
            },
            "output" : []
        }
        self._write_json(empty_log)