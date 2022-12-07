from urllib.request import urlopen
  
import json

TRAIN_DATA = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/train.json"
DEV_DATA = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/dev.json"

url = TRAIN_DATA
response = urlopen(url)
data_json = json.loads(response.read())
print(data_json[0])
