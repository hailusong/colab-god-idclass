from pprint import pprint
import PIL
import numpy as np
# from fastai import *
# from fastai.vision import *
# from torchvision import transforms

TRANSPARENT_COLOR = [0, 0, 0]

def load_pascal_voc_json(json_file):
    with open(json_file) as fi:
        return json.load(fi)

def get_all_tags(pascal_voc_json):
    """ return tag array"""
    return pascal_voc_json['inputTags'].split(',')

def get_all_frames(pascal_voc_json):
    """ return frame index array"""
    return pascal_voc_json['visitedFrames']

def get_bbox_by_fname_tag(pascal_voc_json, fname, tag):
    """ return bbox coordinates by frame name and tag"""
    base = basename(fname)
    base_from_0 = int(base) - 1
    return get_bbox_by_frame_tag(pascal_voc_json, f'{base_from_0}', tag)

def get_bbox_by_frame_tag(pascal_voc_json, frame_index, tag):
    """ return bbox coordinates by frame name and tag"""
    all_tag_info = pascal_voc_json['frames'][f'{frame_index}']
    for tag_info in all_tag_info:
        if tag in tag_info['tags']:
            return (tag_info['x1'], tag_info['y1'], tag_info['x2'], tag_info['y2'])

    raise Exception(f'Cannot find bbox info for tag {tag} in frame {frame_index}')

def get_info_by_frame_tag(pascal_voc_json, frame_index, tag, prefix=None):
    """ return bbox coordinates by frame name and tag"""
    all_tag_info = pascal_voc_json['frames'][f'{frame_index}']
    for tag_info in all_tag_info:
        if tag in tag_info['tags']:
            fname = f'{prefix}/{frame_index+1}'
            fname_JPG = f'{fname}.JPG'
            fname_PNG = f'{fname}.PNG'
            fname_jpg = f'{fname}.jpg'
            fname_png = f'{fname}.png'
            fname = fname_JPG if Path(fname_JPG).is_file() \
                    else fname_PNG if Path(fname_PNG).is_file() \
                    else fname_jpg if Path(fname_jpg).is_file() \
                    else fname_png

            return (fname,
                    tag_info['x1'], tag_info['y1'], tag_info['x2'], tag_info['y2'])

    raise Exception(f'Cannot find info for tag {tag} in frame {frame_index}')

# def merge_image(img_from:Image, img_to:ImageBBox)->ImageBBox:
#     "merge src image to the area of target image as defined in image bbox"
#     x = PIL.Image.open(fn).convert('RGB')
#     return Image(pil2tensor(x,np.float32).div_(255))
#
def crop(img_data, bbox):
    original = image.array_to_img(img_data, scale=True)
    cropped = original.crop(bbox)
    return image.img_to_array(original)

def bbox_size(bbox):
    return (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)

def bbox_position(bbox):
    return (bbox[0], bbox[1])

# def points2bbox(pnts:Collection[torch.Tensor],
#         clip_size:tuple=None)->Collection[Collection[int]]:
#     """
#     find out the bounding box of given points
#
#     pnts:
#         x-first point collection, shape [n, 2]
#
#     clip_size:
#         If specified, clip out-of-bound point to on-the-bound.
#     """
#     if type(pnts) == list:
#         pnts = torch.stack(pnts)
#
#     min_x = pnts[:, 0].min()
#     max_x = pnts[:, 0].max()
#     min_y = pnts[:, 1].min()
#     max_y = pnts[:, 1].max()
#     if clip_size is not None:
#         min_x = max(min_x, 0)
#         min_y = max(min_y, 0)
#         max_x = min(max_x, clip_size[0])
#         max_y = min(max_y, clip_size[1])
#
#     return [min_x, min_y, max_x, max_y]
#
# def valid_offset(pnts:Collection[torch.Tensor])->Tuple:
#     """
#     find out the minimum offsets on x and y respectively so that
#     by applying the offsets will make all x/y points be valid (> 0)
#     """
#     if type(pnts) == list:
#         pnts = torch.stack(pnts)
#     assert(type(pnts) == torch.Tensor)
#
#     min_x = pnts[:, 0].min()
#     min_y = pnts[:, 1].min()
#     offset_x = 0 if min_x >= 0 else -min_x
#     offset_y = 0 if min_y >= 0 else -min_y
#
#     return torch.tensor([offset_x, offset_y]).float()

def resize(img:PIL.Image, size):
    return img.resize(size, PIL.Image.ANTIALIAS)

def thumbnail(img:PIL.Image, size):
    img.thumbnail(size, PIL.Image.ANTIALIAS)
    return img

def new_image(im_shape, convert_mode:str='RGB', pad_color=TRANSPARENT_COLOR):
    return PIL.Image.new(convert_mode, im_shape,
        pad_color if type(pad_color) == Tuple else tuple(pad_color))

def paste(img_bg:PIL.Image, position, img_fg:PIL.Image, img_fg_mask:PIL.Image=None):
    """ paste image_fg to image_bg"""
    if img_fg.mode == 'RGBA':
        # print('!!!!!!!!RGBA!!!!!!!!!!!!')
        img_bg.paste(img_fg, position, mask=img_fg)
    else:
        img_bg.paste(img_fg, position, mask=img_fg_mask)

    return img_bg

def pad(img:PIL.Image, padzone_pct:float):
    """
    Note that this func will add some padding space around the image bg to avoid
    cropping caused by rotation transformation (no-clip). This applies to the
    scenario that we need to rotate composed ID image before merging with
    background image. By adding padding we can make sure the rotation
    will not clip the real ID image itself.

    padzone_pct : float
        Percentage of image width and image height as padding space
    """
    # img_width, img_height = img.size
    assert(type(img.size) == tuple)
    sz_change = tuple((np.multiply(img.size, padzone_pct) / 2).astype(int))
    return pad_by_height(img, sz_change[0], sz_change[1], sz_change[0], sz_change[1])

def pad_by_height(img:PIL.Image, left:int=0, top:int=0, right:int=0, bottom:int=0)->PIL.Image:
    """
    Note that this func will add some padding space around the image bg to avoid
    cropping caused by rotation transformation (no-clip). This applies to the
    scenario that we need to rotate composed ID image before merging with
    background image. By adding padding we can make sure the rotation
    will not clip the real ID image itself.

    left:int, top:int, right:int, bottom:int
        left/top/right/bottom padding height (in pixels)
    """
    # img_width, img_height = img.size
    assert(type(img.size) == tuple)
    im_size_new = tuple(np.add(img.size, (left + right, top + bottom)).astype(int))
    img_pad = new_image(im_size_new)
    paste(img_pad,
          (left, top),
          img)

    return img_pad

def mask(img:PIL.Image, bg_color:list=TRANSPARENT_COLOR)->PIL.Image:
    """
    create image's mask using specified background color.

    note that fast.ai image transformation will automatically create RGBA and
    generate correct image mask. The issue is we pad the image with extra space
    prior to the transformation (so that to avoid clipping) and that is causing
    all padding area considered as image content (not to be masked).
    This function will check pixles one by one and create the mask on pixel's color
    (bg_color or not).

    bg_color:list
        [R, G, B]
    """
    # img_data shape is (height, width, depth)
    img_data = np.array(img)
    # # transpose it to (width, height, depth)
    # img_data = np.transpose(img_data, (1, 0, 2))
    # img size is (wdith, height)
    assert(img_data.shape == (img.size[1], img.size[0], 3)), f'mask input size is {img_data.shape}'
    # bg_color => True
    # not bg_color => False
    # True/false => 255/0
    mask_data = (img_data != bg_color).any(-1).astype('uint8') * 255
    assert(mask_data.shape == (img.size[1], img.size[0])), f'mask size is {mask_data.shape}'

    return PIL.Image.fromarray(mask_data)

def basename(file_path):
    base = os.path.basename(file_path)
    return os.path.splitext(base)[0]

def preview_pil(img:PIL.Image):
    img.show()
    input(f'bbox is {unpad_bbox(img)}')

def unpad_bbox(img:PIL.Image):
    img = img.convert('RGB')
    return img.getbbox()

def unpad_size(img:PIL.Image):
    """
    @return
        (height, width)
    """
    # PIL.Image.getbbox() won't work with RGBA. RGB only.
    bbox = unpad_bbox(img)
    # return [bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]
    return [bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1]

def preview(img_data):
    original = image.array_to_img(img_data, scale=False)
    original.show()

def class_indices_by_index(class_indices, index):
    for key, value in class_indices.items():
        if value == index:
            return key

    raise Exception('Cannot find class by index {index}')

# def fastaiimg2pil(img:Image)->NPArray:
#     imgdata = img.data
#     return tensor2pil(imgdata)
#
# def tensor2pil(tensor:TensorImage)->NPArray:
#     trans = transforms.ToPILImage()
#     return trans(tensor.data)
#
def pil2np(img:PIL.Image):
    return np.array(img)

def open_image_pil(fname:str)->PIL.Image:
    return PIL.Image.open(fname)

# def get_voc_annotations(fname, prefix=None, annotations:Collection[Tuple]=[]):
#     "Open a VOC style json in `fname` and returns the lists of filenames (with maybe `prefix`) and labelled bboxes."
#     json_data = load_pascal_voc_json(fname)
#
#     fns, fn2bboxes, fn2cats = [], collections.defaultdict(list), collections.defaultdict(list)
#
#     # pprint(json_data)
#     tags = get_all_tags(json_data)
#     frames = get_all_frames(json_data)
#     for frameidx in frames:
#         for tag in tags:
#             fn, x1, y1, x2, y2 = get_info_by_frame_tag(json_data, frameidx, tag, prefix)
#             if fn not in fns:
#                 fns.append(fn)
#             fn2bboxes[fn].append([y1, x1, y2, x2])
#             fn2cats[fn].append(tag)
#
#     return fns, fn2bboxes, fn2cats
#
# def get_voc_annotations_fastai(
#     fname, prefix=None, annotations:Collection[Tuple]=[]):
#     fns, fn2bboxes, fn2cats = get_voc_annotations(fname, prefix)
#     lbl_bboxes = [[fn2bboxes[fn], fn2cats[fn]] for fn in fns]
#     return fns, dict(zip(fns, lbl_bboxes))

    # annot_dict = json.load(open(fname))
    # id2images, id2bboxes, id2cats = {}, collections.defaultdict(list), collections.defaultdict(list)
    # classes = {}
    # for o in annot_dict['categories']:
    #     classes[o['id']] = o['name']
    # for o in annot_dict['annotations']:
    #     bb = o['bbox']
    #     id2bboxes[o['image_id']].append([bb[1],bb[0], bb[3]+bb[1], bb[2]+bb[0]])
    #     id2cats[o['image_id']].append(classes[o['category_id']])
    # for o in annot_dict['images']:
    #     if o['id'] in id2bboxes:
    #         id2images[o['id']] = ifnone(prefix, '') + o['file_name']
    # ids = list(id2images.keys())
    # return [id2images[k] for k in ids], [[id2bboxes[k], id2cats[k]] for k in ids]
