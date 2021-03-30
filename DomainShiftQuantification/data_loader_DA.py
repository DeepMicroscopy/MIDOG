import numpy as np
from pathlib import Path
from SlideRunner_dataAccess.database import Database
import openslide

from random import randint

from lib.object_detection_helper import *

from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.data_block import *

from enum import Enum

class DomainDefinition():
    TARGET = 0
    SOURCE = 1
    
class KnownAnnotation():
    UKNOWN_ANNOTATION = 0
    KNOWN_ANNOTATION = 1
    

from fastai.vision.image import _draw_rect

class ImageBBoxWithDomain(ImagePoints):
    "Support applying transforms to a `flow` of bounding boxes."
    def __init__(self, flow:FlowField, scale:bool=True, y_first:bool=True, domain:DomainDefinition=None, labels:Collection=None,
                 classes:dict=None, pad_idx:int=0):
        super().__init__(flow, scale, y_first)
        self.pad_idx = pad_idx
        self.domain = tensor(domain).float()
        if (sum(self.domain)==0):
            print('Created domain==0 image', domain)
            raise ValueError()
        if labels is not None and len(labels)>0 and not isinstance(labels[0],Category):
            labels = array([Category(l,classes[l]) for l in labels])
        self.labels = labels

    def clone(self) -> 'ImageBBoxWithDomain':
        "Mimic the behavior of torch.clone for `Image` objects."
        flow = FlowField(self.size, self.flow.flow.clone())
        return self.__class__(flow, scale=False, y_first=False, domain=self.domain, labels=self.labels, pad_idx=self.pad_idx)

    @classmethod
    def create(cls, h:int, w:int, bboxes:Collection[Collection[int]],  labels:Collection=None, classes:dict=None,
               pad_idx:int=0, scale:bool=True, domain:DomainDefinition=None)->'ImageBBoxWithDomain':
        "Create an ImageBBoxWithDomain object from `bboxes`."
        if isinstance(bboxes, np.ndarray) and bboxes.dtype == np.object: bboxes = np.array([bb for bb in bboxes])
        bboxes = tensor(bboxes).float()
        if (domain is None):
            raise ValueError('Domain not set. Error!')
        tr_corners = torch.cat([bboxes[:,0][:,None], bboxes[:,3][:,None]], 1)
        bl_corners = bboxes[:,1:3].flip(1)
        bboxes = torch.cat([bboxes[:,:2], tr_corners, bl_corners, bboxes[:,2:]], 1)
        flow = FlowField((h,w), bboxes.view(-1,2))
        return cls(flow, labels=labels, classes=classes, pad_idx=pad_idx, y_first=True, scale=scale, domain=domain)

    def _compute_boxes(self) -> Tuple[LongTensor, LongTensor]:
        bboxes = self.flow.flow.flip(1).view(-1, 4, 2).contiguous().clamp(min=-1, max=1)
        mins, maxes = bboxes.min(dim=1)[0], bboxes.max(dim=1)[0]
        bboxes = torch.cat([mins, maxes], 1)
        mask = (bboxes[:,2]-bboxes[:,0] > 0) * (bboxes[:,3]-bboxes[:,1] > 0)
        if len(mask) == 0: return tensor([self.pad_idx] * 4), tensor([self.pad_idx])
        res = bboxes[mask]
        if self.labels is None: return res,None
        return res, self.labels[to_np(mask).astype(bool)]

    @property
    def data(self)->Union[FloatTensor, Tuple[FloatTensor,LongTensor], FloatTensor]:
        bboxes,lbls = self._compute_boxes()
        lbls = np.array([o.data for o in lbls]) if lbls is not None else None
        return (bboxes,self.domain) if lbls is None else (bboxes, lbls, self.domain)

    def show(self, y:Image=None, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True,
        color:str='white', **kwargs):
        "Show the `ImageBBoxWithDomain` on `ax`."
#        print('showing data ..')
        if ax is None: _,ax = plt.subplots(figsize=figsize)
        bboxes, lbls = self._compute_boxes()
        h,w = self.flow.size
        bboxes.add_(1).mul_(torch.tensor([h/2, w/2, h/2, w/2])).long()
        for i, bbox in enumerate(bboxes):
#            print('Index=',i,self.domain)
            if lbls is not None: text = str(lbls[i])
            else: text=None
            _draw_rect(ax, bb2hw(bbox), text=text, color=color)
        ax.set_title(f'Domain {self.domain[0]}')
        

            

class SlideContainer():

    def __init__(self, file: Path, annotations:dict, y, level: int=0, width: int=512, height: int=512, sample_func: callable=None, domain:DomainDefinition=None):
        self.file = file
        self.slide = openslide.open_slide(str(file))
        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]
        self.y = y
        self.annotations = annotations
        self.domain = domain
        self.sample_func = sample_func
        self.classes = list(set(self.y[1]))

        if level is None:
            level = self.slide.level_count - 1
        self.level = level

    def get_patch(self,  x: int=0, y: int=0):
        return np.array(self.slide.read_region(location=(int(x * self.down_factor),int(y * self.down_factor)),
                                          level=self.level, size=(self.width, self.height)))[:, :, :3]

    @property
    def shape(self):
        return (self.width, self.height)

    def __str__(self):
        return 'SlideContainer with:\n sample func: '+str(self.sample_func)+'\n slide:'+str(self.file)

    def get_new_train_coordinates(self):
        # use passed sampling method
        if callable(self.sample_func):
            return self.sample_func(self.y, **{"classes": self.classes, "size": self.shape,
                                               "level_dimensions": self.slide.level_dimensions,
                                               "annotations" : self.annotations,
                                               "level": self.level, "container" : self})

        # use default sampling method
        class_id = np.random.choice(self.classes, 1)[0]
        ids = self.y[1] == class_id
        xmin, ymin, _, _ = np.array(self.y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]
        return int(xmin - self.shape / 2), int(ymin - self.height / 2)

def bb_pad_collate_min(samples:BatchSamples, pad_idx:int=0) -> Tuple[FloatTensor, Tuple[LongTensor, LongTensor]]:
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
#    print('Collate in: ', len(samples), len(samples[0]), samples[0][0].shape, len(samples[0][1].data), len(samples[0][-1].data))

    samples = [s for s in samples if s[1].data[0].shape[0] > 0] # check that labels are available


    max_len = max([len(s[1].data[1]) for s in samples])
    bboxes = torch.zeros(len(samples), max_len, 4)
    labels = torch.zeros(len(samples), max_len).long() + pad_idx
    domains = torch.zeros(len(samples),1, dtype=torch.long)
    #print('Domains: ', [s[1].data[2] for s in samples ])
    imgs = []
    for i,s in enumerate(samples):
        imgs.append(s[0].data[None])
        bbs, lbls, domain = s[1].data
        bboxes[i,-len(lbls):] = bbs
        labels[i,-len(lbls):] = torch.from_numpy(lbls)
        try:
            if len(domain.shape)==0:
                domains[i,0] = domain
            else:
                domains[i,0] = domain[0]
        except Exception as e:
            print('Domain is: ',domain.shape,domain)
            print('Index is:',i)
            print(s[1].data)
            print('Error is:',e)
            if isinstance(lbls,list):
                print('lbls is list of length', len(lbls))
            else:
                print(lbls.shape)
            
            raise(e)
    #print('Shapes: Domains:',domains.shape,'Bboxes:', labels.shape, 'BBoxes:',bboxes.shape)
    #print('Out domains:',domains)
    #print('Collate result: ', bboxes.shape, labels.shape, domains.shape)
    return torch.cat(imgs,0), (bboxes,labels,domains)

class SlideLabelList(LabelList):


    def __getitem__(self,idxs:Union[int,np.ndarray])->'LabelList':
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            if self.item is None:
                slide_container = self.x.items[idxs]

                xmin, ymin = slide_container.get_new_train_coordinates()

                x = self.x.get(idxs, xmin, ymin)
                y = self.y.get(idxs, xmin, ymin)
                
            else:
                x,y = self.item ,0
            if self.tfms or self.tfmargs:
                x = x.apply_tfms(self.tfms, **self.tfmargs)
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve':False})
            if y is None: y=0
            return x,y
        else:
            return self.new(self.x[idxs], self.y[idxs])



PreProcessors = Union[PreProcessor, Collection[PreProcessor]]
fastai_types[PreProcessors] = 'PreProcessors'

class SlideItemList(ItemList):

    def __init__(self, items:Iterator, path:PathOrStr='.', label_cls:Callable=None, inner_df:Any=None,
                 processor:PreProcessors=None, x:'ItemList'=None, ignore_empty:bool=False):
        self.path = Path(path)
        self.num_parts = len(self.path.parts)
        self.items,self.x,self.ignore_empty = items,x,ignore_empty
        self.sizes = [None] * len(self.items)
        if not isinstance(self.items,np.ndarray): self.items = array(self.items, dtype=object)
        self.label_cls,self.inner_df,self.processor = ifnone(label_cls,self._label_cls),inner_df,processor
        self._label_list,self._split = SlideLabelList,ItemLists
        self.copy_new = ['x', 'label_cls', 'path']

    def __getitem__(self,idxs: int, x: int=0, y: int=0)->Any:

        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            return self.get(idxs, x, y)
        else:
            return self.get(*idxs)

    def label_from_list(self, labels:Iterator, label_cls:Callable=None, **kwargs)->'LabelList':
        "Label `self.items` with `labels`."

        labels = array(labels, dtype=object)
        label_cls = self.get_label_cls(labels, label_cls=label_cls, **kwargs)
        y = label_cls(labels, path=self.path,  **kwargs)
        res = SlideLabelList(x=self, y=y)
        return res


class SlideImageItemList(SlideItemList):
    pass

class SlideObjectItemList(SlideImageItemList, ImageList):

    def get(self, i, x: int, y: int):

        fn = self.items[i]
        res = self.open(fn, x, y)
        self.sizes[i] = res.size
        return res

class ObjectItemListSlide(SlideObjectItemList):

    def open(self, fn: SlideContainer,  x: int=0, y: int=0):
        return Image(pil2tensor(fn.get_patch(x, y) / 255., np.float32))

    
from fastai.vision.data import ObjectCategoryProcessor

class SlideObjectCategoryProcessor(MultiCategoryProcessor):
    "`PreProcessor` for labelled bounding boxes."
    def __init__(self, ds:ItemList, pad_idx:int=0):
        super().__init__(ds)
        self.pad_idx = pad_idx
        self.state_attrs.append('pad_idx')

    def process(self, ds:ItemList):
        ds.pad_idx = self.pad_idx
        super().process(ds)

    def process_one(self,item): return [item[0], [self.c2i.get(o,None) for o in item[1]], item[2]]

    def generate_classes(self, items):
        "Generate classes from unique `items` and add `background`."
        classes = super().generate_classes([o[1] for o in items])
        classes = ['background'] + list(classes)
        return classes

class SlideObjectCategoryList(MultiCategoryList):
    _processor = SlideObjectCategoryProcessor

    def __init__(self, items:Iterator, classes:Collection=None, label_delim:str=None, one_hot:bool=False, **kwargs):
        if label_delim is not None: items = array(csv.reader(items.astype(str), delimiter=label_delim))
        super().__init__(items, classes=classes, **kwargs)
        if one_hot:
            assert classes is not None, "Please provide class names with `classes=...`"
            self.processor = [MultiCategoryProcessor(self, one_hot=True)]
        self.loss_func = BCEWithLogitsFlat()
        self.one_hot = one_hot
        for k in range(len(items)):
            try:
                assert(len(items[k])==3)
            except Exception as e:
                print('Failed at: ',k)
                raise(e)
        self.copy_new += ['one_hot']
        self.items = items
        
#    def get(self, i):
#        return ImageBBox.create(*_get_size(self.x,i), *self.items[i], classes=self.classes, pad_idx=self.pad_idx)

    def analyze_pred(self, pred): return pred

    def reconstruct(self, t, x):
        (bboxes, labels, domain) = t
        if len((labels - self.pad_idx).nonzero()) == 0: 
            labels = np.array([0])
            bboxes = np.array([[-1, -1, 1, 1]])
            return ImageBBoxWithDomain.create(*x.size, bboxes, labels=labels, domain=domain, classes=self.classes, scale=False)
        i = (labels - self.pad_idx).nonzero().min()
        bboxes,labels = bboxes[i:],labels[i:]
        return ImageBBoxWithDomain.create(*x.size, bboxes, labels=labels, domain=domain, classes=self.classes, scale=False)    


    def get(self, i, x: int=0, y: int=0):

        h, w = self.x.items[i].shape

        bboxes, labels, domain = self.items[i]
        if len(labels)>0 and x > 0 and y > 0:
            bboxes = np.array(bboxes)
            labels = np.array(labels)
            domain = np.array(domain[0:1])

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - y

            bb_widths = (bboxes[:, 2] - bboxes[:, 0]) / 2
            bb_heights = (bboxes[:, 3] - bboxes[:, 1]) / 2

            ids = ((bboxes[:, 0] + bb_widths) > 0) \
                  & ((bboxes[:, 1] + bb_heights) > 0) \
                  & ((bboxes[:, 2] - bb_widths) < w) \
                  & ((bboxes[:, 3] - bb_heights) < h)

            bboxes = bboxes[ids]
            bboxes = np.clip(bboxes, 0, x)
            bboxes = bboxes[:, [1, 0, 3, 2]]

            labels = labels[ids]
            if len(labels) == 0:
                labels = np.array([0])
                bboxes = np.array([[0, 0, 1, 1]])
            return ImageBBoxWithDomain.create(h, w, bboxes, labels, classes=self.classes, domain=domain, pad_idx=self.pad_idx)
        else:
            if len(labels) == 0:
                labels = np.array([0])
                bboxes = np.array([[0, 0, 1, 1]])
            if len(domain)==0:
                print('Dataset that caused error was: ',self.items[i],i)
                raise ValueError('No domain given')
            return ImageBBoxWithDomain.create(h, w, bboxes[:10], labels[:10], classes=self.classes, domain=domain, pad_idx=self.pad_idx)


def slide_object_result(learn: Learner, anchors, detect_thresh:float=0.2, nms_thresh: float=0.3,  image_count: int=5):
    with torch.no_grad():
        img_batch, target_batch = learn.data.one_batch(DatasetType.Train, False, False, False)
        prediction_batch = learn.model(img_batch)
        class_pred_batch, bbox_pred_batch = prediction_batch[:2]

        bbox_gt_batch, class_gt_batch, domain_batch = target_batch

        for img, bbox_gt, class_gt, clas_pred, bbox_pred, domain_gt in \
                list(zip(img_batch, bbox_gt_batch, class_gt_batch, class_pred_batch, bbox_pred_batch, domain_batch))[:image_count]:
            img = Image(learn.data.denorm(img))

            out = process_output(clas_pred, bbox_pred, anchors, detect_thresh)
            bbox_pred, scores, preds = [out[k] for k in ['bbox_pred', 'scores', 'preds']]
            if bbox_pred is not None:
                to_keep = nms(bbox_pred, scores, nms_thresh)
                bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()

            t_sz = torch.Tensor([*img.size])[None].cpu()
            bbox_gt = bbox_gt[np.nonzero(class_gt)].squeeze(dim=1).cpu()
            class_gt = class_gt[class_gt > 0] - 1
            # change gt from x,y,x2,y2 -> x,y,w,h
            bbox_gt[:, 2:] = bbox_gt[:, 2:] - bbox_gt[:, :2]

            bbox_gt = to_np(rescale_boxes(bbox_gt, t_sz))
            if bbox_pred is not None:
                bbox_pred = to_np(rescale_boxes(bbox_pred, t_sz))
                # change from center to top left
                bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2

            show_results(img, bbox_pred, preds, scores, list(range(0, learn.data.c))
                         , bbox_gt, class_gt, (15, 3), titleA=f'Domain {domain_gt}', titleB=str(''), titleC='CAM', clas_pred=clas_pred, anchors=anchors)


def show_results_with_breg(img, bbox_pred, preds, scores, breg_pred, classes, bbox_gt, preds_gt, figsize=(5,5)
                 , titleA: str="", titleB: str="", titleC: str=""):

    _, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    ax[0].set_title(titleA)
    ax[1].set_title(titleB)
    ax[2].set_title(titleC)

    # show gt
    img.show(ax=ax[0])
    for bbox, c in zip(bbox_gt, preds_gt):
        txt = str(c.item()) if classes is None else classes[c.item()]
        draw_rect(ax[0], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt}')

    # show prediction class
    img.show(ax=ax[1])
    if bbox_pred is not None:
        for bbox, c, scr in zip(bbox_pred, preds, scores):
            txt = str(c.item()) if classes is None else classes[c.item()]
            draw_rect(ax[1], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt} {scr.item():.1f}')

    # show prediction class
    img.show(ax=ax[2])
    if bbox_pred is not None:
        for bbox, c in zip(bbox_pred, breg_pred):
            draw_rect(ax[1], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{c.item():.1f}')


