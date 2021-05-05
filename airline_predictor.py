import requests 
import json 

url = "https://airline-satisfaction-app.herokuapp.com/predict?att0=Male&att1=Loyal+Customer&att2=1&att3=Business+travel&att4=Business&att5=1"


response = requests.get(url=url)

status_code = response.status_code
print("status_code:", status_code)
print("message body:", response.text)

if status_code == 200:
    json_object = json.loads(response.text)
    print(json_object)