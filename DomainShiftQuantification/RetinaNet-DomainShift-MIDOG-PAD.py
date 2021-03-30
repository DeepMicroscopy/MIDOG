#!/usr/bin/env python
#
#  Training of a domain classifier from a RetinaNet-implementation to approximate the domain gap (via proxy A distance)
#

# Run as: python3 RetinaNet-DomainShift-MIDOG-PAD.py <scanner_source_domain> <scanner_target_domain> <run>
#
# where run is a number indicating the training run of the original RetinaNet

import os
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
from data_loader_DA import *
from lib.callbacks import BBLossMetrics, BBMetrics, PascalVOCMetric
from lib.object_detection_helper import *
from model.RetinaNetFocalLoss import RetinaNetFocalLoss
from model.RetinaNet import RetinaNet
from lib.sampling import get_slides
size=512
path = Path('./')

database = Database()
database.open(str('MIDOG.sqlite'))

scanner_indomain_selected = int(sys.argv[1]) # 1:XR, 2: S360, 3:CS2, 4: GT450
scanner_outdomain_selected = int(sys.argv[2]) # 1:XR, 2: S360, 3:CS2, 4: GT450
run = int(sys.argv[3])

import random
random.seed(42)
slides = [int(x[0]) for x in database.execute('SELECT uid from Slides').fetchall()]
random.shuffle(slides)


fold = 0
slidelist_test = [str(x) for x in np.arange(41,51).tolist()+np.arange(91,101).tolist()+np.arange(141,151).tolist()+np.arange(191,201).tolist()]
slidelist_trainval = []

scanner_slidelists = { 1: np.arange(1,41).tolist(),
                       2: np.arange(51,91).tolist(),
                       3: np.arange(101,141).tolist(),
                       4: np.arange(151,191).tolist()}
slidelist_trainval = scanner_slidelists[scanner_indomain_selected]+scanner_slidelists[scanner_outdomain_selected]
random.shuffle(slidelist_trainval)
slidelist_val = slidelist_trainval[0:10]

print('Test slides are: ',slidelist_test)
print('Val slides are: ',slidelist_val)


# In[2]:


def sampling_func(y, **kwargs):
#    print(y[1])
    try:
        y_label = np.array(y[1])
    except:
        y_label = y[1]
    h, w = kwargs['size']
    _arbitrary_prob = 0.1
    _mit_prob = 0.5
    
    sample_prob = [_arbitrary_prob, 1-_arbitrary_prob-_mit_prob, _mit_prob]
        
    
    case = np.random.choice(3, p=sample_prob)
    
    
    
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
        xmin, ymin = randint(int(w - xoffset), slide_width - w), randint(int(h- yoffset), int(slide_height) - h)
        
        
    if (case==2): # mitosis
        
        ids = y_label == 1

        if (np.count_nonzero(ids)>0):
            if (_set == 'training'):
                ids[coords[:,1]>slide_height/2] = 0 # lower part not allowed
            elif (_set == 'validation'):
                ids[coords[:,1]<slide_height/2] = 0 # upper part not allowed

        if (np.count_nonzero(ids)<1):
            xmin, ymin = randint(int(w - xoffset), slide_width - w), randint(int(h- yoffset), int(slide_height) - h)
        else:
            xmin, ymin, xmax, ymax = np.array(y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]
           # print('Mitosis: ', xmin,ymin,xmax,ymax )
    if (case==1): #nonmitosis
            annos = kwargs['annotations']
            if 2 not in annos:
                xmin, ymin = randint(int(w - xoffset), slide_width - w), randint(int(h- yoffset), int(slide_height) - h)
            else:
                try:
                    coords = np.array(annos[2]['bboxes'])
    #        That did not work {2: {'bboxes': [[3110, 1703, 3160, 1753], [3427, 2863, 3477, 2913], [4878, 3380, 4928, 3430], [5042, 4199, 5092, 4249], [4396, 4641, 4446, 4691]], 'label': [2, 2, 2, 2, 2]}}
                except:
                    print('That did not work', annos)
                    raise ValueError()
            
                try:
                    ids = np.arange(len(coords))
                except:
                    print(coords)
                    raise ValueError('')

                if (np.count_nonzero(ids)<1):
                    xmin, ymin = randint(int(w - xoffset), slide_width - w), randint(int(h- yoffset), int(slide_height) - h)

                else:
                    xmin, ymin, xmax, ymax = coords[ids][randint(0, np.count_nonzero(ids) - 1)]

    return int(xmin - w / 2 + xoffset), int(ymin - h / 2 +yoffset)


# In[5]:


database = Database()
database.open(str('MIDOG.sqlite'))

lbl_bbox=list()
files=list()
train_slides=list()
val_slides=list()
test_slides=list()
getslides = """SELECT uid, filename FROM Slides"""
for idx, (currslide, filename) in enumerate(tqdm(database.execute(getslides).fetchall(), desc='Loading slides .. ')):
    if (str(currslide) in slidelist_test): # skip test slides
        continue
    domain = int((currslide-1) / 50)+1
    if (domain==scanner_indomain_selected):
        domain=1
    elif (domain==scanner_outdomain_selected):
        domain=2
    else:
        continue
    
    database.loadIntoMemory(currslide)

    slide_path = path / 'images_training' / filename

    slide = openslide.open_slide(str(slide_path))
    
    level = 0#slide.level_count - 1
    level_dimension = slide.level_dimensions[level]
    down_factor = slide.level_downsamples[level]

    classes = {1: 1} # Map non-mitosis to background
#    classes = {0: 'unknown', 1: 'Non-Mitosis', 2: 'Mitosis'}

    labels, bboxes = [], []
    annotations = dict()
    for id, annotation in database.annotations.items():
        annotation.r = 25
        if annotation.deleted or annotation.annotationType==AnnotationType.POLYGON:
            continue
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

    labelsfound = max(1,len(labels))
    lbl_bbox.append([bboxes, labels, [domain,]*labelsfound])
    files.append(SlideContainer(file=slide_path, domain=domain,  annotations=annotations, level=level, width=size, height=size, y=[bboxes, labels, [domain,]*labelsfound], sample_func=partial(sampling_func, level_dimensions=level_dimension, set='training')))
    if currslide not in slidelist_val:
        train_slides.append(len(files)-1)
    else:
        val_slides.append(len(files)-1)


            


# ## Split dataset into train/validation and test on slide level

# In[8]:


len(val_slides), len(train_slides), len(files)


# In[9]:


img2bbox = dict(zip(files, np.array(lbl_bbox)))
get_y_func = lambda o:img2bbox[o]


# In[10]:


# transfer learn from TUPAC_AL
fname = 'RetinaNet-TUPAC_AL-OrigSplit-512s.pth'
if (scanner_indomain_selected==1):
    fname = f'RetinaNet-MIDOG-XR-{run}.pth'
elif (scanner_indomain_selected==2):
    fname = f'RetinaNet-MIDOG-S360-{run}.pth'
elif (scanner_indomain_selected==3):
    fname = f'RetinaNet-MIDOG-CS2-{run}.pth'
    
    
state = torch.load(fname, map_location='cpu')     if defaults.device == torch.device('cpu')     else torch.load(fname)
model = state.pop('model').cuda()
mean = state['data']['normalize']['mean']
std = state['data']['normalize']['std']


# In[ ]:





# In[ ]:





# In[11]:


bs = 20
train_images = 5000
val_images = 5000

train_files = list(np.random.choice([files[x] for x in val_slides], val_images))
valid_files = list(np.random.choice([files[x] for x in val_slides], val_images))

#train_files = list(np.random.choice(files, train_images))
#valid_files = list(np.random.choice(files, val_images))


# In[12]:


anchors = create_anchors(sizes=[(32,32)], ratios=[1], scales=[0.6, 0.7,0.8,0.9])


# In[13]:


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
train =  ObjectItemListSlide(train_files, path=path)
valid = ObjectItemListSlide(valid_files, path=path)
valid = ObjectItemListSlide(valid_files, path=path)
item_list = ItemLists(path, train, valid)
lls = item_list.label_from_func(get_y_func, label_cls=SlideObjectCategoryList) #
lls = lls.transform(tfms, tfm_y=True, size=size)
data = lls.databunch(bs=bs, collate_fn=bb_pad_collate_min, num_workers=4).normalize()


# In[78]:


#data.show_batch(rows=3)


# In[24]:


class ProxyANetwork(nn.Module):
    def __init__(self, RetinaNetModel, num_classes=4):
        super().__init__()
        for p in RetinaNetModel.parameters():
            p.requres_grad=False
        self.encoder = RetinaNetModel.encoder
        self.model = RetinaNetModel
        self.conv1 = conv2d(512, 512, ks=3, bias=True).cuda()
        self.conv2 = conv2d(512, 512, ks=3, stride=2, bias=True).cuda()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3 = conv2d(512,num_classes,ks=1).cuda()
        self.domainClass_ENC = nn.Sequential(self.conv1,self.conv2, nn.ReLU(), self.avgpool, self.conv3)

        self.cconv1 = conv2d(128, 512, ks=3, bias=True).cuda()
        self.cconv2 = conv2d(512, 512, ks=3, stride=2, bias=True).cuda()
        self.cavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cconv3 = conv2d(512,num_classes,ks=1).cuda()
        self.domainClass_CLA = nn.Sequential(self.cconv1,self.cconv2, nn.ReLU(), self.cavgpool, self.cconv3)

        self.bconv1 = conv2d(128, 512, ks=3, bias=True).cuda()
        self.bconv2 = conv2d(512, 512, ks=3, stride=2, bias=True).cuda()
        self.bavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bconv3 = conv2d(512,num_classes,ks=1).cuda()
        self.domainClass_BOX = nn.Sequential(self.bconv1,self.bconv2, nn.ReLU(), self.bavgpool, self.bconv3)
        
        
    def forward(self,x):
        c5 = self.encoder(x)
        p_states = [self.model.c5top5(c5.clone()), self.model.c5top6(c5)]
        p_states.append(self.model.p6top7(p_states[-1]))
      #  print('Got ',x.shape[0],'images')
        for merge in self.model.merges:
            p_states = [merge(p_states[0])] + p_states
        for i, smooth in enumerate(self.model.smoothers[:3]):
            p_states[i] = smooth(p_states[i])
        if self.model.sizes is not None:
            p_states = [p_state for p_state in p_states if p_state.size()[-1] in self.model.sizes]
        feats = []
        feats_rev = []

        stat = torch.cat(p_states)
        for fwdlayer in list(self.model.classifier.children())[:-1]:
            stat = fwdlayer(stat)

        de = self.domainClass_ENC(c5)
       # print('Predicted',de.shape[0],'domains')
        retval=[de,self.domainClass_CLA(stat)]
        stat = torch.cat(p_states)
        for fwdlayer in list(self.model.box_regressor.children())[:-1]:
            stat = fwdlayer(stat)
        
        retval.append(self.domainClass_BOX(stat))
        return retval


# In[25]:


mdl = ProxyANetwork(model,num_classes=2)


# In[ ]:





# In[26]:


from torch.autograd import Variable

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

class DomainLoss(nn.Module):

    def __init__(self, anchors, classes):
        super().__init__()
        self.loss_domain = torch.nn.NLLLoss()
        self.anchors = anchors
        self.classes = classes
        self.pad_idx = 0
        self.metric_names = ['EncLoss', 'ClaLoss', 'BoxLoss']

    def _unpad(self, bbox_tgt, clas_tgt):
        try:
            i = torch.min(torch.nonzero(clas_tgt - self.pad_idx))
        except:
            i = 0
        return tlbr2cthw(bbox_tgt[i:]), clas_tgt[i:] - 1 + self.pad_idx


    def _check_use(self, clas_tgt, bbox_tgt):
        bbox_tgt, clas_tgt = self._unpad(bbox_tgt, clas_tgt)
        matches = match_anchors(self.anchors.to(bbox_tgt.device), bbox_tgt)
        # filter out those regression predictions where annotations are available
        bbox_mask = (matches >= 0)
        if bbox_mask.sum() != 0:
            return True
        else:
            bb_loss = 0.
            return False
    
    def forward(self, output, bbox_tgts, clas_tgts, domain_tgts):
        domain1,domain2,domain3 = output
        domain_tgts=domain_tgts-torch.ones(domain_tgts.shape).to(domain_tgts.device)
        domain_tgts = to_one_hot(domain_tgts, self.classes).to(domain2.device)[:,0,:]
        confusion_loss = F.binary_cross_entropy_with_logits(domain2[:,:,0,0], domain_tgts, reduction='mean')
        
        return confusion_loss



crit=DomainLoss(anchors, 2)


# In[29]:


learn = Learner(data, mdl, loss_func=crit, callback_fns=[ShowGraph], #BBMetrics, ShowGraph
               )


# In[30]:


for name,x in learn.model.named_parameters():
    if 'model' in name or 'encoder' in name:
        x.requires_grad=False


# In[31]:


[(name,x.requires_grad) for name,x in list(learn.model.named_parameters())]


# In[32]:


#learn.lr_find()
#learn.recorder.plot()


# In[33]:


learn.fit(1,1e-4)


# In[34]:


learn.fit_one_cycle(10,1e-4)

learn.fit(30, 1e-4, callbacks=[SaveModelCallback(learn, every='improvement', name='model')])


# In[35]:


#learn.save('RetinaNet_PAD_{scanner_indomain_selected}_{scanner_outdomain_selected}')


# In[36]:


slidedir = 'images_training'
test_files = []
for idx, (currslide, filename) in enumerate(tqdm(database.execute(getslides).fetchall(), desc='Loading slides .. ')):
        if str(currslide) not in slidelist_test:
            continue
        
        domain = int((currslide-1) / 50)+1
        
        if (domain not in [scanner_indomain_selected,scanner_outdomain_selected]):
            continue
            

        database.loadIntoMemory(currslide)

        slide_path = path / slidedir / filename
        scont = SlideContainer(file=slide_path, level=level, width=size, height=size, y=[[], []], annotations=dict())
        test_files.append(scont)


# In[71]:


from queue import Queue
import queue
import torchvision.transforms as transforms
import threading
jobQueue=Queue()
outputQueue=Queue()

def getPatchesFromQueue(jobQueue, outputQueue):
    x,y=0,0
    try:
        while (True):
            if (outputQueue.qsize()<100):
                status, x,y, slide_container = jobQueue.get(timeout=60)
                if (status==-1):
                    return
                outputQueue.put((x,y,slide_container.get_patch(x, y) / 255.))
            else:
                time.sleep(0.1)
    except queue.Empty:
        print('One worker died.')
        pass # Timeout happened, exit



def getBatchFromQueue(batchsize=8):
    images = np.zeros((batchsize,3, size,size))
    x = np.zeros(batchsize)
    y = np.zeros(batchsize)
    try:
        bs=0
        for k in range(batchsize):
            x[k],y[k],images_temp = outputQueue.get(timeout=5)
            images[k] = images_temp.transpose((2,0,1))
            bs+=1
        return images,x,y
    except queue.Empty:
        return images[0:bs],x[0:bs],y[0:bs]
    
# Set up queued image retrieval
jobs = []
for i in range(1):
    p = threading.Thread(target=getPatchesFromQueue, args=(jobQueue, outputQueue), daemon=True)
    jobs.append(p)
    p.start()


# In[73]:


def inference(files, model, mean, std, batchsize=8):
    elements=0
    error=0
    with torch.no_grad():
        for slide_container in tqdm(files):

            size = 512


            n_Images=0
            for x in range(0, slide_container.slide.level_dimensions[level][0] - 1 * size, int(0.9*size)):
                for y in range(0, slide_container.slide.level_dimensions[level][1] - 1*  size, int(0.9*size)):
                    jobQueue.put((0,x,y, slide_container))
                    n_Images+=1

            domain=int((int(str(slide_container.file.name.split('.')[0]))-1)/50)+1
            print('Target domain: ',domain)
            dtarget = 0 if domain==scanner_indomain_selected else 1

            for kImage in range(int(np.ceil(n_Images/batchsize))):


                    npBatch,xBatch,yBatch = getBatchFromQueue(batchsize=batchsize)
                    imageBatch = torch.from_numpy(npBatch.astype(np.float32, copy=False)).cuda()

                    patch = imageBatch

                    for p in range(patch.shape[0]):
                        patch[p] = transforms.Normalize(mean,std)(patch[p])

                    domain1, domain2, domain3 = model(
                        patch[:, :, :, :])
                    
                    
                    
                    for b in range(patch.shape[0]):
                        error += abs(domain2.argmax(1)[b]-dtarget) 
                        elements += 1.0
                    
#                    print(class_pred_batch.argmax(1).shape)
                        
            print('Error:',error, 'Elements',elements, 'MAE',error/elements)
    return error/elements


# In[74]:


model = learn.model.eval()
MAE = inference(test_files,model,*data.stats)


# In[87]:


pickle.dump(1-2*MAE.cpu().numpy()[0][0],open(f'padstar_clahead_{scanner_indomain_selected}_{scanner_outdomain_selected}_{run}.p','wb'))

print(f'MAE for configuration: {scanner_indomain_selected}_{scanner_outdomain_selected}_{run} ==> ', MAE.cpu().numpy()[0][0])
