import configparser
from google.cloud import videointelligence_v1p3beta1 as videointelligence
import imageio
from PIL import Image, ImageDraw
import numpy
import pickle
import tqdm
import os
from urllib.parse import urlparse

def detect_faces(gcs_uri):
    """
    TODO
    """    
    # """Detects faces in a video."""

    # client = videointelligence.VideoIntelligenceServiceClient()

    # # Configure the request
    # config = videointelligence.types.FaceDetectionConfig(
    #     include_bounding_boxes=True, include_attributes=True
    # )
    # context = videointelligence.types.VideoContext(face_detection_config=config)

    # # Start the asynchronous request
    # operation = client.annotate_video(
    #     input_uri=gcs_uri,
    #     features=[videointelligence.enums.Feature.FACE_DETECTION],
    #     video_context=context,
    # )

    # print("\nProcessing video for face detection annotations.")
    # result = operation.result(timeout=300)

    # print("\nFinished processing.\n")

    # res_file = open("res_file.dict", "wb")
    # pickle.dump(result, res_file)
    # res_file.close()
    res_file = open("res_file.dict", "rb")
    result = pickle.load(res_file)

    return result

def generate_offset_dict(result):
    """
    TODO
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

def draw_boundingboxes(image, boundingboxes):
    """
    TODO
    """
    h_res, v_res = image.size
    draw = ImageDraw.Draw(image)
    for i, boundingbox in enumerate(boundingboxes['boundingbox']):
        left = boundingbox.left * h_res
        top = boundingbox.top * v_res
        right = boundingbox.right * h_res
        bottom = boundingbox.bottom * v_res

        draw.rectangle([left ,top, right, bottom])
        for i, attribute in enumerate(boundingboxes['attributes'][i]):
            if(attribute.confidence >= 0.5):
                draw.text([left,bottom+5*i], attribute.name)

    return numpy.array(image)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    #image io constructs
    input_file_name = os.path.basename(urlparse(config['global']['input_video_gcs_uri']).path)
    reader = imageio.get_reader('news_test.mp4')
    print(reader.get_meta_data())
    fps = float(config['global']['fps'])

    writer = imageio.get_writer(config['global']['output_video_filename'], fps=fps)

    result = detect_faces(config['global']['input_video_gcs_uri'])

    offset_dict = generate_offset_dict(result)

    for i, im in enumerate(reader):
        frame_offset = "%.6f" % (i/fps)
        print('Timestamp of frame %i is %s' % (i, frame_offset))
        if frame_offset in offset_dict:
            bounded_image = draw_boundingboxes(Image.fromarray(im), offset_dict[frame_offset])     
            writer.append_data(bounded_image)
        else:
            writer.append_data(im)

        if i == 200:
            break

    writer.close()