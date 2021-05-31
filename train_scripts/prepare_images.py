from operator import itemgetter

from transformer import TRTPoseExtractor
import cv2
import os
import pandas as pd
import argparse

video_dir = ''
image_dir = ''
csv_path = ''
frame_interval = 5

extractor = TRTPoseExtractor()


def get_bbox(kp_list):
    bbox = []
    for aggs in [min, max]:
        for idx in range(2):
            bound = aggs(kp_list, key=itemgetter(idx))[idx]
            bbox.append(bound)
    return bbox


def extract_images(video_path, parsed_data_arr):
    cap = cv2.VideoCapture('{0}'.format(video_path))
    # Check if camera opened successfully
    filename = video_path.split('/')[-1]
    if not cap.isOpened():
        print("Error opening video stream or file")
    frame_num = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            if frame_num % frame_interval == 0:
                print('inference frame {0} from file {1}'.format(frame_num, video_path))
                sample = extractor.transform([frame])
                print(sample)
                for idx, res in enumerate(sample):
                    try:
                        bbox = get_bbox(list(res[0].values()))
                        rect_x1_f, rect_y1_f, rect_x2_f, rect_y2_f = bbox
                        person = frame[rect_y1_f:rect_y2_f, rect_x1_f:rect_x2_f]
                        imgname = image_dir + filename + '_' + str(frame_num) + '_' + str(idx) + '.jpg';
                        cv2.imwrite(imgname, person)
                        print(imgname)
                        parsed_data_arr.append([imgname, res[1], class_label])
                    except:
                        print('no person found')
                if frame_num % 100 == 0:
                    df = pd.DataFrame(parsed_data_arr, columns=['image', 'vec', 'label'])
                    df.to_csv(csv_path, encoding='utf-8', index=False)
            frame_num += 1
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--class_label", required=True)
    args = parser.parse_args()

    video_dir = args.video_folder
    image_dir = args.output_folder
    csv_path = args.output_csv
    class_label = args.class_label

    for subdir, dirs, files in os.walk(video_dir):
        parsed_data = []
        for video in files:
            video_src = os.path.join(subdir, video)
            extract_images(video_src, parsed_data)
