import imageio
from PIL import Image, ImageDraw
import numpy
import json
import objectpath

json_data = json.load(open('response.json'))
json_tree = objectpath.Tree(json_data)


def get_boundingboxes_by_frame(frame_offset):
    """
    TODO
    """
    return json_tree.execute("$..*[@.timeOffset is '{}s']".format(frame_offset))
    
def draw_boundingboxes(image, boundingboxes):
    """
    TODO
    """
    draw = ImageDraw.Draw(image)

    for boundingbox in boundingboxes:
        print(boundingbox)
        try:
            left = boundingbox['normalizedBoundingBox']['left'] * 1280
        except:
            left = 0
        
        try: 
            top = boundingbox['normalizedBoundingBox']['top'] * 544
        except: 
            top = 0

        try:
            right = boundingbox['normalizedBoundingBox']['right'] * 1280
        except:
            right = 1280

        try:
            bottom = boundingbox['normalizedBoundingBox']['bottom'] * 544
        except:
            bottom = 544

        # print('BBoxes: %.5f %.5f %.5f %.5f' % (left, top, right, bottom))

        draw.rectangle([left ,top, right, bottom])
        # draw.text([5,10], "Hola mundo")
        print("exit")

    return numpy.array(image)

if __name__ == "__main__":

    #image io constructs
    reader = imageio.get_reader('goldeneye.mp4')
    print(reader.get_meta_data())
    #Seems not to work properly - fps = reader.get_meta_data()['fps']
    fps = 23.976024

    writer = imageio.get_writer('imageio.mp4', fps=fps)

    for i, im in enumerate(reader):
        print('Timestamp of frame %i is %.6f' % (i, i/fps))

        frame_offset = "%.6f" % (i/fps)
        bounding_boxes = get_boundingboxes_by_frame(frame_offset)
  
        bounded_image = draw_boundingboxes(Image.fromarray(im), bounding_boxes)
     
        writer.append_data(bounded_image)

        # if i == 50:
        #     break

    writer.close()

