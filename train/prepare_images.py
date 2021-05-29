from operator import itemgetter

from transformer import TRTPoseExtractor, GetKeypoints
import cv2
import os

video_dir = './dataset/fight'
image_dir = './dataset/images/fight'

extractor = TRTPoseExtractor()

def get_bbox(kp_list):
    bbox = []
    for aggs in [min, max]:
        for idx in range(2):
            bound = aggs(kp_list, key=itemgetter(idx))[idx]
            bbox.append(bound)
    return bbox

def extract_images(video_src):
    cap = cv2.VideoCapture(video_src)
    # Check if camera opened successfully
    filename = video_src.split('/')[-1]
    if not cap.isOpened():
        print("Error opening video stream or file")
    frame_num = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            sample = extractor.transform([frame])
            for idx, body in enumerate(sample):
                bbox = get_bbox(list(body.values()))
                rect_x1_f, rect_y1_f, rect_x2_f, rect_y2_f = bbox
                person = frame[rect_y1_f:rect_y2_f, rect_x1_f:rect_x2_f]
                cv2.imwrite(filename+'_'+str(frame_num)+'_'+str(idx)+'.jpg', person)

            frame_num += 1
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()


for subdir, dirs, files in os.walk(video_dir):
    for video in files:
        video_src = os.path.join(subdir, video)
        extract_images(video_src)
        # label = subdir.split('/')[-1]
        # labeled_images.append([image_path, label])

# image = cv2.imread('/your/sample/image.jpg')
# sample = extractor.transform([image])
# print(type(sample))
# print(sample.shape)
# print(sample)
