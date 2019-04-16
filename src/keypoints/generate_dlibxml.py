import fire
import glob
import pandas as pd

from utils import *

BBOX_PREFIX = 'bbox-'
PNTS_PREFIX = 'pnts-'

class DlibGenerator:
    def _generate(self, bbox_file:str, pnts_file:str, img_prefix:str=None)->str:
        print(f'working on {bbox_file} and {pnts_file}')
        bbox_pd = pd.read_csv(bbox_file).set_index('Unnamed: 0')
        pnts_pd = pd.read_csv(pnts_file).set_index('Unnamed: 0')
        assert(bbox_pd.shape[0] == pnts_pd.shape[0]), f'{bbox_pd.shape} != {pnts_pd.shape}'

        xml_buff = []
        for index, bbox_row in bbox_pd.iterrows():
            # return is [top_shift, left_shift, bottom_shift, right_shift]
            shifts = random_shift(bbox_row)

            # update bbox_row
            bbox_row['bbox1_y1'] -= shifts[0]
            bbox_row['bbox1_x1'] -= shifts[1]
            bbox_row['bbox1_y2'] += shifts[2]
            bbox_row['bbox1_x2'] += shifts[3]

            # bbox_row has been updated by adding some randomness
            pnts_row = pnts_pd.loc[index, :]
            xml = to_img_xml(index, bbox_row, pnts_row, img_prefix)
            xml_buff.extend(xml)

        return xml_buff


    def generate(self, bboxes_csv:str, xml_fn:str, img_prefix:str=None):
        """
        generate dlib training/validation xml data based on bboxes and points CSV data.
        note that this function will infer points data filename (pnts-*.csv) from bboxes data filename (bbox-*.csv).
        this function will add some randomness to the bboxes for data augmentation purpose.

        bboxes_csv : str
            Input bboxes csv filename (support wildcard)

        pnts_csv : str
            Input key points csv filename (support wildcard)

        xml_fn : str
            output XML filename

        img_prefix : str
            prefix should be inserted at the beginning of all image file pathes
        """
        xml_buffer = []

        for bbox_file in glob.glob(bboxes_csv, recursive=False):
            pnts_file = bbox_file.replace(BBOX_PREFIX, PNTS_PREFIX)
            xml_buffer.extend(self._generate(bbox_file, pnts_file, img_prefix))

        full_xml = to_full_xml('idclass', f'data is for {xml_fn}', xml_buffer)

        with open(xml_fn, 'w') as of:
            of.write(full_xml)

        print(f'Payload XML count: {len(xml_buffer)}, total image count: {len(xml_buffer)/6}')


def main():
    # as of 2018/12, no Windows multi-processing support
    # torch.multiprocessing.freeze_support()
    fire.Fire(DlibGenerator())


if __name__ == '__main__':
    # as of 2018/12, no Windows multi-processing support
    # pool = multiprocessing.Pool(4)
    # torch.multiprocessing.freeze_support()
    # multiprocessing.freeze_support()
    main()
