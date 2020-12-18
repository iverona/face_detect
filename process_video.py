import configparser
from google.cloud import videointelligence_v1p3beta1 as videointelligence
import cv2
import numpy
import pickle
import os
from urllib.parse import urlparse

def detect_faces(gcs_uri):
    """
    Detects faces in a video.
    """
    result = ""

    input_file_name = os.path.basename(urlparse(gcs_uri).path)
    save_to_file = "%s.res" % input_file_name

    if os.path.isfile(save_to_file):
        print("Found existing file. Using it!")
        res_file = open(save_to_file, "rb")
        result = pickle.load(res_file)        
    else:
        client = videointelligence.VideoIntelligenceServiceClient()

        # Configure the request
        config = videointelligence.types.FaceDetectionConfig(
            include_bounding_boxes=True, include_attributes=True
        )
        context = videointelligence.types.VideoContext(face_detection_config=config)

        # Start the asynchronous request
        operation = client.annotate_video(
            input_uri=gcs_uri,
            features=[videointelligence.enums.Feature.FACE_DETECTION],
            video_context=context,
        )

        print("\nProcessing video for face detection annotations.")
        result = operation.result(timeout=600)

        print("\nFinished processing.\n")

        res_file = open(save_to_file, "wb")
        pickle.dump(result, res_file)
        res_file.close()

    return result
    

def generate_offset_dict(result):
    """
    Transform the API response to be addressable
    by frame offset. It will return a dictionary 
    with the offset as the key and boundingbox and 
    attributes as lists. 
    """
    offset_dict = {}
    annotation_result = result.annotation_results[0]
    for annotation in annotation_result.face_detection_annotations:
        for track in annotation.tracks:
            for tso in track.timestamped_objects:
                key = "%.6f" % (tso.time_offset.seconds + tso.time_offset.nanos / 1e9)
                boundingbox = tso.normalized_bounding_box
                attributes = tso.attributes
                if key not in offset_dict:
                    offset_dict[key] = { 'boundingbox': [ boundingbox ], 'attributes': [attributes] }
                else:
                    offset_dict[key]['boundingbox'].append(boundingbox)
                    offset_dict[key]['attributes'].append(attributes)

    return offset_dict

def draw_boundingboxes(image, this_frame_boundingboxes):
    """
    Draw bounding boxes and attributes. 
    The attributes are sorted so they do not move
    around in the image so much. 
    """
    v_res, h_res, c = image.shape

    for i, boundingbox in enumerate(this_frame_boundingboxes['boundingbox']):        
        left = int(boundingbox.left * h_res)
        top = int(boundingbox.top * v_res)
        right = int(boundingbox.right * h_res)
        bottom = int(boundingbox.bottom * v_res)
        
        cv2.rectangle(image, (left,top), (right,bottom), (0,255,0), 2)

        sorted_attributes = []
        for attribute in this_frame_boundingboxes['attributes'][i]:
            if(attribute.confidence >= 0.6):
                sorted_attributes.append(attribute.name)

        sorted_attributes.sort()
        
        if sorted_attributes:          
            for i, attribute in enumerate(sorted_attributes):
                    cv2.putText(image, attribute, (left, (bottom+20)+20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    input_file_name = os.path.basename(urlparse(config['global']['input_video_gcs_uri']).path)
    cap = cv2.VideoCapture(input_file_name)
    
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print("File: %s FPS: %f Size: %ix%i" % (input_file_name, fps, frame_width, frame_height))

    out = cv2.VideoWriter(config['global']['output_video_filename'],fourcc, 10, (frame_width,frame_height))

    result = detect_faces(config['global']['input_video_gcs_uri'])

    offset_dict = generate_offset_dict(result)
    f_since_last_hit = 0 # Frames processed since last written frame

    while(cap.isOpened()):
        #Read a frame
        ret, frame = cap.read()
        if ret == True:
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) 
            frame_offset = "%.6f" % (float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000)
            print('Timestamp of frame %i is %s >> ' % (frame_number, frame_offset), end='')
            if frame_offset in offset_dict:
                print('Hit. Appending')
                bounded_image = draw_boundingboxes(frame, offset_dict[frame_offset])     
                out.write(frame)
                f_since_last_hit = 0
            else:
                if f_since_last_hit <= 2:
                    f_since_last_hit += 1
                    print('Skipping')
                    continue
                else:
                    print('Appending')
                    f_since_last_hit = 0
                    out.write(frame)
        else:
            break 

    cap.release()
    out.release()
