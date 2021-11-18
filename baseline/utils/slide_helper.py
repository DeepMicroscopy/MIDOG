import os
import sys
sys.path.append(os.path.abspath('../../SlideRunner'))
from SlideRunner.dataAccess.database import Database
from object_detection_fastai.helper.wsi_loader import SlideContainer
import numpy as np
from random import *
import json




def sample_function(y, classes, size, level_dimensions, level):
    width, height = level_dimensions[level]
    if len(y[0]) == 0:
        xmin, ymin = randint(0, width - size[0]), randint(0, height - size[1])
    else:
        if randint(0,5) < 3:
            class_id = np.random.choice(classes, 1)[0]
            ids = np.array(y[1]) == class_id
            xmin, ymin, _, _ = np.array(y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]
            xmin -= randint(0,size[0])
            ymin -= randint(0,size[1])
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmin, ymin = min(xmin, width - size[0]), min(ymin, height - size[1])
        else:
            xmin, ymin = randint(0, width - size[0]), randint(0, height - size[1])
    return xmin, ymin


def load_images(slide_folder, annotation_file, res_level, patch_size, scanner_id, categories):
    container = []
    anno_dict = {1: "mitotic figure", 2: "impostor"}
    for image in os.listdir(slide_folder):
        if annotation_file.split(".")[-1] == "json":
            with open(annotation_file) as f:
                data = json.load(f)
                image_id = [i["id"] for i in data["images"] if i["file_name"] == image][0]
                annotations = [anno for anno in data['annotations'] if anno["image_id"] == image_id and anno["category_id"] in categories]
                bboxes = [a["bbox"] for a in annotations]
                labels = [anno_dict[a["category_id"]] for a in annotations]
                container.append(SlideContainer(os.path.join(slide_folder, image), y=[bboxes, labels, scanner_id], level=res_level, width=patch_size, height=patch_size, sample_func=sample_function))
        elif annotation_file.split(".")[-1] == "sqlite":
            DB = Database().open(annotation_file)
            slideid = DB.findSlideWithFilename(image, '')
            DB.loadIntoMemory(slideid)
            bboxes = [DB.annotations[anno].coordinates.flatten() for anno in DB.annotations.keys() if
                      DB.annotations[anno].deleted == 0]
            labels = [DB.annotations[anno].agreedClass for anno in DB.annotations.keys() if
                      DB.annotations[anno].deleted == 0]
            container.append(SlideContainer(os.path.join(slide_folder, image), y=[bboxes, labels], level=res_level,
                                            width=patch_size, height=patch_size))
        else:
            print("Please provide valid annotation format")
    return container