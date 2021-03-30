# Cross-Domain Training of a model on the MIDOG mitosis domain generalization challenge
#
# Authors: M. Aubreville, C. Marzahl
# 



import sqlite3
import numpy as np
from SlideRunner_dataAccess.database import Database
from SlideRunner_dataAccess.annotations import AnnotationType
from tqdm import tqdm
from pathlib import Path
import openslide
import time
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from random import randint
from data_loader import *
from lib.callbacks import BBLossMetrics, BBMetrics, PascalVOCMetric
from lib.object_detection_helper import *
from model.RetinaNetFocalLoss import RetinaNetFocalLoss
from model.RetinaNet import RetinaNet
from lib.sampling import get_slides
size=512
path = Path('./')

database = Database()
database.open(str('MIDOG.sqlite'))

import random
random.seed(42)
slides = [int(x[0]) for x in database.execute('SELECT uid from Slides').fetchall()]
random.shuffle(slides)

from sklearn.model_selection import KFold
folds = KFold(3)
f = list(folds.split(slides))
sets = [np.array(slides)[x[1]] for x in f]
trainsets = [np.array(slides)[x[0]] for x in f]
import sys
scannerTestset = {'XR': np.arange(41,201),
     'S360': np.arange(1,51).tolist()+np.arange(91,201).tolist(),
     'CS2' : np.arange(1,101).tolist()+np.arange(141,201).tolist()}

slidelist_test = [str(x) for x in scannerTestset[sys.argv[1]]]
    
slidelist_trainval = [x for x in np.arange(1,201) if str(x) not in slidelist_test]
fold = 0


# In[30]:


slidelist_val = random.sample(slidelist_trainval,10)


# In[31]:


slidelist_trainval


# ## Split dataset into train/validation and test on slide level

# Convert database into in-memory object

# In[32]:


def sampling_func(y, **kwargs):
    y_label = np.array(y[1])
    h, w = kwargs['size']

    _arbitrary_prob = 0.1
    _mit_prob = 0.9
    
    sample_prob = np.array([_arbitrary_prob,  _mit_prob])
    
    case = np.random.choice(2, p=sample_prob)
    
    
    
    bg_label = [0] if y_label.dtype == np.int64 else ["bg"]
    classes = bg_label + kwargs['classes']
    level_dimensions = kwargs['level_dimensions']
    level = kwargs['level']
    if ('bg_label_prob' in kwargs):
        _bg_label_prob = kwargs['bg_label_prob']
        if (_bg_label_prob>1.0):
            raise ValueError('Probability needs to be <= 1.0.')
    else:
        _bg_label_prob = 0.0  # add a backgound label to sample complete random
    
    if ('strategy' in kwargs):
        _strategy = kwargs['strategy']
    else:
        _strategy = 'normal'
        
    if ('set' in kwargs):
        _set = kwargs['set']
    else:
        _set = 'training'

        
    _random_offset_scale = 0.5  # up to 50% offset to left and right of frame
    xoffset = randint(-w, w) * _random_offset_scale
    yoffset = randint(-h, h) * _random_offset_scale
    coords = np.array(y[0])

    slide_width, slide_height = level_dimensions[level]
    
    if (case==0):
        xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), slide_height - h)
    if (case==1): # mitosis
        
        ids = y_label == 1

        if (_set == 'training'):
            ids[coords[:,1]>slide_height/2] = 0 # lower part not allowed
        elif (_set == 'validation'):
            ids[coords[:,1]<slide_height/2] = 0 # upper part not allowed

        if (np.count_nonzero(ids)<1):
            xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), slide_height - h)
        else:
            xmin, ymin, xmax, ymax = np.array(y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]

    return int(xmin - w / 2 + xoffset), int(ymin - h / 2 +yoffset)


def get_slides(slidelist_test:list, database:"Database", positive_class:int=1, negative_class:int=7, basepath:str='WSI', slidelist_val:list=[], size:int=256):


    lbl_bbox=list()
    files=list()
    train_slides=list()
    val_slides=list()

    getslides = """SELECT uid, directory,filename FROM Slides"""
    for idx, (currslide, direct, filename) in enumerate(tqdm(database.execute(getslides).fetchall(), desc='Loading slides .. ')):
        if (str(currslide) in slidelist_test): # skip test slides
            continue

        database.loadIntoMemory(currslide)

        slide_path = basepath + os.sep + filename

        slide = openslide.open_slide(str(slide_path))

        level = 0#slide.level_count - 1
        level_dimension = slide.level_dimensions[level]
        down_factor = slide.level_downsamples[level]

        classes = {positive_class: 1} # Map non-mitosis to background

        labels, bboxes = [], []
        annotations = dict()
        for id, annotation in database.annotations.items():
            if annotation.deleted or annotation.annotationType != AnnotationType.SPOT:
                continue
            annotation.r = 25
            d = 2 * annotation.r / down_factor
            x_min = (annotation.x1 - annotation.r) / down_factor
            y_min = (annotation.y1 - annotation.r) / down_factor
            x_max = x_min + d
            y_max = y_min + d
            if annotation.agreedClass not in annotations:
                annotations[annotation.agreedClass] = dict()
                annotations[annotation.agreedClass]['bboxes'] = list()
                annotations[annotation.agreedClass]['label'] = list()

            annotations[annotation.agreedClass]['bboxes'].append([int(x_min), int(y_min), int(x_max), int(y_max)])
            annotations[annotation.agreedClass]['label'].append(annotation.agreedClass)

            if annotation.agreedClass in classes:
                label = classes[annotation.agreedClass]

                bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                labels.append(label)

        if len(bboxes) > 0:
            if (len(slidelist_val)==0):
                lbl_bbox.append([bboxes, labels])
                files.append(SlideContainer(file=slide_path, annotations=annotations, level=level, width=size, height=size, y=[bboxes, labels], sample_func=partial(sampling_func, set='training', negative_class=negative_class)))
                train_slides.append(len(files)-1)

                lbl_bbox.append([bboxes, labels])
                files.append(SlideContainer(file=slide_path, annotations=annotations, level=level, width=size, height=size, y=[bboxes, labels], sample_func=partial(sampling_func, set='validation', negative_class=negative_class)))
                val_slides.append(len(files)-1)
            else:
                if (currslide not in slidelist_val):
                    lbl_bbox.append([bboxes, labels])
                    files.append(SlideContainer(file=slide_path, annotations=annotations, level=level, width=size, height=size, y=[bboxes, labels], sample_func=partial(sampling_func, set='training', negative_class=negative_class)))
                    train_slides.append(len(files)-1)
                else:
                    lbl_bbox.append([bboxes, labels])
                    files.append(SlideContainer(file=slide_path, annotations=annotations, level=level, width=size, height=size, y=[bboxes, labels], sample_func=partial(sampling_func, set='validation', negative_class=negative_class)))
                    val_slides.append(len(files)-1)

    return lbl_bbox, train_slides,val_slides,files


# In[33]:


lbl_bbox, train_slides,val_slides,files = get_slides(slidelist_test=slidelist_test, slidelist_val=slidelist_val, negative_class=2, positive_class=1,  
                                                     size=size,database=database,basepath='./images_training/')


# In[34]:


img2bbox = dict(zip(files, np.array(lbl_bbox)))
get_y_func = lambda o:img2bbox[o]


# In[35]:


sorted(slidelist_val), sorted(slidelist_trainval), sorted(slidelist_test)


# In[36]:


train_slides


# In[37]:


bs = 12
train_images = 5000
val_images = 5000

train_files = list(np.random.choice([files[x] for x in train_slides], train_images))
valid_files = list(np.random.choice([files[x] for x in val_slides], val_images))

#train_files = list(np.random.choice(files, train_images))
#valid_files = list(np.random.choice(files, val_images))


# In[38]:


tfms = get_transforms(do_flip=True,
                      flip_vert=True,
                      max_rotate=90,
                      max_lighting=0.0,
                      max_zoom=1.2,
                      max_warp=0.0,
                      p_affine=0.5,
                      p_lighting=0.2,
                      #xtra_tfms=xtra_tfms,
                     )


# In[39]:


path


# In[40]:


train =  ObjectItemListSlide(train_files, path=path)
valid = ObjectItemListSlide(valid_files, path=path)
valid = ObjectItemListSlide(valid_files, path=path)
item_list = ItemLists(path, train, valid)
lls = item_list.label_from_func(get_y_func, label_cls=SlideObjectCategoryList) #
lls = lls.transform(tfms, tfm_y=True, size=size)
data = lls.databunch(bs=bs, collate_fn=bb_pad_collate_min, num_workers=4).normalize()


# In[41]:


tfms


# In[42]:


data.show_batch(rows=2, ds_type=DatasetType.Train, figsize=(15,15))


# In[13]:


anchors = create_anchors(sizes=[(32,32)], ratios=[1], scales=[0.6, 0.7,0.8,0.9])


# In[14]:


not_found = show_anchors_on_images(data, anchors)


# In[ ]:





# In[15]:


crit = RetinaNetFocalLoss(anchors)


# In[16]:


encoder = create_body(models.resnet18, True, -2)
model = RetinaNet(encoder, n_classes=data.train_ds.c, n_anchors=4, sizes=[32], chs=128, final_bias=-4., n_conv=3)


# In[17]:


# transfer learn from TUPAC_AL
fname = 'RetinaNet-TUPAC_AL-OrigSplit-512s.pth'
state = torch.load(fname, map_location='cpu')     if defaults.device == torch.device('cpu')     else torch.load(fname)
model = state.pop('model').cuda()
mean = state['data']['normalize']['mean']
std = state['data']['normalize']['std']


# In[18]:


voc = PascalVOCMetric(anchors, size, [str(i-1) for i in data.train_ds.y.classes[1:]])
learn = Learner(data, model, loss_func=crit, callback_fns=[BBMetrics, ShowGraph], #BBMetrics, ShowGraph
                metrics=[voc]
               )


# In[19]:


# Transfer learning from TUPAC_AL


# In[20]:


learn.split([model.encoder[6], model.c5top5])
learn.freeze_to(-2)


# In[22]:


learn.fit_one_cycle(1, 1e-5)


# In[23]:


lr=1e-5
learn.fit_one_cycle(10, lr)#, callbacks=[SaveModelCallback(learn, every='improvement', monitor='AP-0', name='model')])


# In[ ]:





# In[24]:


learn.unfreeze()
lr=1e-4

learn.fit(30, lr, callbacks=[SaveModelCallback(learn, every='improvement', monitor='pascal_voc_metric', name='model')])


# In[26]:


#learn.save('RetinaNet-TUPAC_AL-OrigSplit-512s', with_opt=True)
learn.export(f"RetinaNet-MIDOG-{sys.argv[1]}-{sys.argv[2]}.pth")


# In[27]:


#torch.save(learn.model.state_dict(), "RetinaNet-TUPAC_CB-512s-OrigSplit_statedict.pth")

