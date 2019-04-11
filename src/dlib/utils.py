
import pandas as pd
import random


XML_PROLOGUE = \
    f"<?xml version='1.0' encoding='ISO-8859-1'?>\n" + \
    "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n" + \
    "<dataset>\n" + \
    "<name>{0}</name>\n" + \
    "<comment>{1}</comment>\n" + \
    "<images>"

XML_EPILOGUE = \
    f"</images>\n" + \
    "</dataset>"

XML_IMAGE_PROLOGUE = \
    "  <image file='{0}'>\n" + \
    "    <box top='{1}' left='{2}' width='{3}' height='{4}'>"

XML_IMAGE_EPILOGUE = "    </box>\n  </image>"

XML_PNTS = "      <part name='{0}' x='{1}' y='{2}'/>"


def bbox_to_xml(img_fn:str, bbox_row):
    """
        bbox_row is Dataframe row
        {
        'top': bbox_row['bbox1_y1'],
        'left': bbox_row['bbox1_x1'],
        'width': bbox_row['bbox1_x2'] - bbox_row['bbox1_x1'] + 1,
        'height': bbox_row['bbox1_y2'] - bbox_row['bbox1_y1'] + 1
        }
    """
    return XML_IMAGE_PROLOGUE.format(
            img_fn,
            bbox_row['bbox1_y1'],
            bbox_row['bbox1_x1'],
            bbox_row['bbox1_x2'] - bbox_row['bbox1_x1'] + 1,
            bbox_row['bbox1_y2'] - bbox_row['bbox1_y1'] + 1
        )


def pnts_to_xml(bbox_row, pnts_row):
    """ pnts_row is Dataframe row with columns x1,y1,x2,y2,x3,y3,x4,y4 """
    base_x, base_y = bbox_row['bbox1_x1'], bbox_row['bbox1_y1']

    return [
        XML_PNTS.format(0, pnts_row['x1'] - base_x, pnts_row['y1'] - base_y),
        XML_PNTS.format(1, pnts_row['x2'] - base_x, pnts_row['y2'] - base_y),
        XML_PNTS.format(2, pnts_row['x3'] - base_x, pnts_row['y3'] - base_y),
        XML_PNTS.format(3, pnts_row['x4'] - base_x, pnts_row['y4'] - base_y)
    ]


def to_img_xml(index:str, bbox_row, pnts_row)->str:
    bbox_xml = bbox_to_xml(index, bbox_row)
    pnts_xml = pnts_to_xml(bbox_row, pnts_row)
    return [bbox_xml, *pnts_xml, XML_IMAGE_EPILOGUE]


def to_full_xml(name:str, comment:str, xml_buff:list)->str:
    return '\n'.join([XML_PROLOGUE.format(name, comment), *xml_buff, XML_EPILOGUE])


def random_shift(bbox_row):
    """randomly shift/size bbox a little bit to augment the data further"""
    top, left = bbox_row['bbox1_y1'], bbox_row['bbox1_x1']
    bottom, right = bbox_row['bbox1_y2'], bbox_row['bbox1_x2']
    img_height, img_width = 256, 256
    assert(img_height > top >= 0 and
           img_width > left >= 0 and
           img_height > bottom >= 0  and
           img_width > right >= 0), \
           f'Out-of-range: {top}/{left}, {bottom}/{right} not within {img_height}/{img_width}'

    random_max = 32
    random_val1 = random.randint(0, random_max)
    random_val2 = random.randint(0, random_max)
    random_val3 = random.randint(0, random_max)
    random_val4 = random.randint(0, random_max)

    top = max(0, top - random_val1)
    left = max(0, left - random_val2)
    bottom = min(img_height, bottom + random_val3)
    right = min(img_width, right + random_val4)

    bbox_row['bbox1_y1'] = top
    bbox_row['bbox1_x1'] = left
    bbox_row['bbox1_y2'] = bottom
    bbox_row['bbox1_x2'] = right

    return bbox_row

# create new method for pd.DataFrame
# pd.DataFrame.to_xml = to_xml
