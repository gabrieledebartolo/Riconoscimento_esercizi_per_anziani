'''
s1_get_skeletons_from_training_imgs.py:
  openpose:
    model: cmu # cmu or mobilenet_thin. "cmu" is more accurate but slower.
    img_size: 656x368 #  656x368, or 432x368, 336x288. Bigger is more accurate.
  input:
    train_images_description_txt: data/training_set/valid_images.txt
    test_images_description_txt: data/test_set/valid_images.txt
    images_train_folder: &train_set_folder data/training_set/
    images_test_folder: &test_set_folder data/test_set/
  output:
    train_images_info_txt: data_proc/raw_skeletons/training_set/images_info.txt
    test_images_info_txt: data_proc/raw_skeletons/test_set/images_info.txt
    detected_skeletons_train_folder: &skels_train_folder data_proc/raw_skeletons/training_set/skeleton_res/
    detected_skeletons_test_folder: &skels_test_folder data_proc/raw_skeletons/test_set/skeleton_res/
    viz_imgs_train_folder: data_proc/raw_skeletons/training_set/image_viz/
    viz_imgs_test_folder: data_proc/raw_skeletons/test_set/image_viz/
'''

import cv2
import yaml

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_skeletons_io import ReadValidImagesAndActionTypesByTxt
    import utils.lib_commons as lib_commons


def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings


cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s1_get_skeletons_from_training_testing_imgs.py"]

IMG_FILENAME_FORMAT = cfg_all["image_filename_format"]
SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]


# Input
if True:
    SRC_TRAIN_IMAGES_DESCRIPTION_TXT = par(cfg["input"]["train_images_description_txt"])
    SRC_TRAIN_IMAGES_FOLDER = par(cfg["input"]["images_train_folder"])
    SRC_TEST_IMAGES_DESCRIPTION_TXT = par(cfg["input"]["test_images_description_txt"])
    SRC_TEST_IMAGES_FOLDER = par(cfg["input"]["images_test_folder"])

# Output
if True:
    DST_TRAIN_FOLDER = par(cfg["output"]["train_folder"])
    DST_TEST_FOLDER = par(cfg["output"]["test_folder"])

    # This txt will store image info, such as index, action label, filename, etc.
    # This file is saved but not used.
    DST_TRAIN_IMAGES_INFO_TXT = par(cfg["output"]["train_images_info_txt"])
    DST_TEST_IMAGES_INFO_TXT = par(cfg["output"]["test_images_info_txt"])

    # Each txt will store the skeleton of each image
    DST_DETECTED_SKELETONS_TRAIN_FOLDER = par(
        cfg["output"]["detected_skeletons_train_folder"])
    DST_DETECTED_SKELETONS_TEST_FOLDER = par(
        cfg["output"]["detected_skeletons_test_folder"])

    # Each image is drawn with the detected skeleton
    DST_VIZ_IMGS_TRAIN_FOLDER = par(cfg["output"]["viz_imgs_train_folder"])
    DST_VIZ_IMGS_TEST_FOLDER = par(cfg["output"]["viz_imgs_test_folder"])

# Openpose
if True:
    OPENPOSE_MODEL = cfg["openpose"]["model"]
    OPENPOSE_IMG_SIZE = cfg["openpose"]["img_size"]

# -- Functions

class ImageDisplayer(object):
    ''' A simple wrapper of using cv2.imshow to display image '''

    def __init__(self):
        self._window_name = "cv2_display_window"
        cv2.namedWindow(self._window_name)

    def display(self, image, wait_key_ms=1):
        cv2.imshow(self._window_name, image)
        cv2.waitKey(wait_key_ms)

    def __del__(self):
        cv2.destroyWindow(self._window_name)


# -- Main
if __name__ == "__main__":

    # -- Image reader and displayer for TRAINING
    images_loader = ReadValidImagesAndActionTypesByTxt(
        img_folder=SRC_TRAIN_IMAGES_FOLDER,
        valid_imgs_txt=SRC_TRAIN_IMAGES_DESCRIPTION_TXT,
        img_filename_format=IMG_FILENAME_FORMAT)
    # This file is not used.
    images_loader.save_images_info(filepath=DST_TRAIN_IMAGES_INFO_TXT)

    # -- Init output path for TRAINING
    os.makedirs(DST_TRAIN_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(DST_TRAIN_IMAGES_INFO_TXT), exist_ok=True)
    os.makedirs(DST_DETECTED_SKELETONS_TRAIN_FOLDER, exist_ok=True)
    os.makedirs(DST_VIZ_IMGS_TRAIN_FOLDER, exist_ok=True)

    try:
        process_images(images_loader, DST_DETECTED_SKELETONS_TRAIN_FOLDER, 
            DST_VIZ_IMGS_TRAIN_FOLDER)
    except:
        print("Exception during training phase")

    # -- Image reader and displayer for TEST
    images_loader = ReadValidImagesAndActionTypesByTxt(
        img_folder=SRC_TEST_IMAGES_FOLDER,
        valid_imgs_txt=SRC_TEST_IMAGES_DESCRIPTION_TXT,
        img_filename_format=IMG_FILENAME_FORMAT)
    # This file is not used.
    images_loader.save_images_info(filepath=DST_TEST_IMAGES_INFO_TXT)

    # -- Init output path for TEST
    os.makedirs(DST_TEST_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(DST_TEST_IMAGES_INFO_TXT), exist_ok=True)
    os.makedirs(DST_DETECTED_SKELETONS_TEST_FOLDER, exist_ok=True)
    os.makedirs(DST_VIZ_IMGS_TEST_FOLDER, exist_ok=True)

    try:
        process_images(images_loader, DST_DETECTED_SKELETONS_TEST_FOLDER, 
            DST_VIZ_IMGS_TEST_FOLDER)
    except:
        print("Exception during test phase")
   
    print("Program ends")

def process_images(images_loader, DST_DETECTED_SKELETONS_FOLDER, DST_VIZ_IMGS_FOLDER):

    # -- Detector
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    multiperson_tracker = Tracker()

    # -- Read images and process
    num_total_images = images_loader.num_images
    for ith_img in range(num_total_images):

        # -- Read image
        img, str_action_label, img_info = images_loader.read_image()

        # -- Detect
        humans = skeleton_detector.detect(img)

        # -- Draw
        img_displayer = ImageDisplayer()
        img_disp = img.copy()
        skeleton_detector.draw(img_disp, humans)
        img_displayer.display(img_disp, wait_key_ms=1)

        # -- Get skeleton data and save to file
        skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
        dict_id2skeleton = multiperson_tracker.track(
            skeletons)  # dict: (int human id) -> (np.array() skeleton)
        skels_to_save = [img_info + skeleton.tolist()
                         for skeleton in dict_id2skeleton.values()]

        # -- Save result

        # Save skeleton data for training
        filename = SKELETON_FILENAME_FORMAT.format(ith_img)
        lib_commons.save_listlist(
            DST_DETECTED_SKELETONS_FOLDER + filename,
            skels_to_save)

        # Save the visualized image for debug
        filename = IMG_FILENAME_FORMAT.format(ith_img)
        cv2.imwrite(
            DST_VIZ_IMGS_FOLDER + filename,
            img_disp)

        print(f"{ith_img}/{num_total_images} th image "
              f"has {len(skeletons)} people in it")
