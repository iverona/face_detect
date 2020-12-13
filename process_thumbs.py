import json
import base64

with open('response.json') as json_file:
    data = json.load(json_file)
    
    counter = 0
    for i in data['response']['annotationResults'][0]['faceDetectionAnnotations']:
        base64_img = i['thumbnail']
        base64_bytes = base64_img.encode('ascii')
        with open('thumbs/img{0}.jpeg'.format(counter), 'wb') as img:
            img.write(base64.decodebytes(base64_bytes))
        counter+=1

