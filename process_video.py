import configparser
from google.cloud import videointelligence_v1p3beta1 as videointelligence
import imageio
from PIL import Image, ImageDraw
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
    Transform the API response to be addressable
    by offset
    """
    h_res, v_res = image.size

    draw = ImageDraw.Draw(image)
    for i, boundingbox in enumerate(this_frame_boundingboxes['boundingbox']):        
        left = boundingbox.left * h_res
        top = boundingbox.top * v_res
        right = boundingbox.right * h_res
        bottom = boundingbox.bottom * v_res

        sorted_attributes = []
        draw.rectangle([left ,top, right, bottom])

        for attribute in this_frame_boundingboxes['attributes'][i]:
            if(attribute.confidence >= 0.6):
                sorted_attributes.append(attribute.name)

        sorted_attributes.sort()
        
        if sorted_attributes:          
            for i, attribute in enumerate(sorted_attributes):
                    draw.text([left,bottom+10*i], attribute)

    return numpy.array(image)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    #image io constructs
    input_file_name = os.path.basename(urlparse(config['global']['input_video_gcs_uri']).path)
    reader = imageio.get_reader(input_file_name)
    print(reader.get_meta_data())
    fps = float(config['global']['fps'])

    writer = imageio.get_writer(config['global']['output_video_filename'], fps=10)

    result = detect_faces(config['global']['input_video_gcs_uri'])

    offset_dict = generate_offset_dict(result)
    f_since_last_hit = 0 # Frames processed since last written frame

    for i, im in enumerate(reader):
        frame_offset = "%.6f" % (i/fps)
        print('Timestamp of frame %i is %s >> ' % (i, frame_offset), end='')
        if frame_offset in offset_dict:
            print('Hit. Appending')
            bounded_image = draw_boundingboxes(Image.fromarray(im), offset_dict[frame_offset])     
            writer.append_data(bounded_image)
            f_since_last_hit = 0
        else:
            if f_since_last_hit <= 2:
                f_since_last_hit += 1
                print('Skipping')
                continue
            else:
                print('Appending')
                f_since_last_hit = 0
                writer.append_data(im)

        # if i == 200:
        #     break

    writer.close()
