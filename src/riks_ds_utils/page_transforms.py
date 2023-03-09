from glob import glob
from pathlib import Path
import cv2
import os
import xml.etree.ElementTree as ET
import json


class PageTransforms:

    def _gather_xmls_imgs(page_path: str, imgs_path: str):

        print(page_path)

        xmls = glob(os.path.join(page_path, '**'))
        imgs = glob(os.path.join(imgs_path, '**'))
        xmls = [x for x in xmls if os.path.isfile(x)]
        imgs = [x for x in imgs if os.path.isfile(x)]
        print(len(imgs))
        print(len(xmls))
        xmls.sort()
        imgs.sort()

        assert len(xmls) == len(imgs)

        return (xmls, imgs)

    def _get_img_shape(img_path: str):
        
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        return (height, width)

    def _get_bbox_area_poly_from_coords(child):

        coordinates = child.attrib['points'].split()
        temp = [coord.split(',') for coord in coordinates]
        temp = [[int(x) + 0.5 for x in lst] for lst in temp]
        px = [x for (x, y) in temp]
        py = [y for (x, y) in temp]
        poly = [p for x in temp for p in x]
                        
        x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))
        bbox=[x_min, y_min, x_max - x_min, y_max - y_min]

        area=(x_max - x_min) * (y_max - y_min)

        return (bbox, area, poly)

    def _write_json(output_path:str, dataset:dict):
        
        with open(output_path, 'w', encoding='utf8') as f:
            json_str = json.dumps(dataset, indent = 4, ensure_ascii=False)
            f.write(json_str)

    

    def page_to_coco(page_path:str, imgs_path:str, out_path:str, elems:list, schema:str=''):
        """_summary_

        Args:
            page_path (str): _description_
            imgs_path (str): _description_
            out_path (str): _description_
            elems (list): _description_
            schema (str, optional): _description_. Defaults to ''.
        """
     
        xmls, imgs = PageTransforms._gather_xmls_imgs(page_path, imgs_path)

        coco = dict()
        coco_annot = list()
        coco_imgs = list()
        coco_categories = list()
        obj_count = 0
        no_of_lines = 0

        for i, el in enumerate(elems):
            coco_categories.append(dict(
                id=i,
                name=el
            ))

        for idx, (image, page) in enumerate(zip(imgs, xmls)):
            
            tree = ET.parse(page)
            root = tree.getroot()

            filename = Path(image).name

            height, width = PageTransforms._get_img_shape(image)

            coco_imgs.append(dict(
                    id=idx,
                    file_name=filename,
                    height=height,
                    width=width
                ))

            schema_formatted = '{' + schema + '}'

            for i, el in enumerate(elems):
                
                for elem in root.iter(schema_formatted + el):
                    no_of_lines += 1

                    for child in elem:

                        if child.tag == schema_formatted + 'Coords':

                            bbox, area, poly = PageTransforms._get_bbox_area_poly_from_coords(child)
                            
                            data_anno = dict(
                                image_id=idx,
                                id=obj_count,
                                category_id=i,
                                bbox=bbox,
                                area=area,
                                segmentation=[poly],
                                iscrowd=0)

                            coco_annot.append(data_anno)

        coco['images'] = coco_imgs
        coco['annotations'] = coco_annot
        coco['categories'] = coco_categories

        PageTransforms._write_json('/home/erik/Riksarkivet/Projects/riks_ds_utils/data/processed/test_coco.json', coco)
            
        

    def page_to_mmlabs_ocr(page_path:str, imgs_path:str, out_path:str, elem_type:str = 'TextLine', schema:str=''):
        """_summary_

        Args:
            page_path (str): _description_
            imgs_path (str): _description_
            out_path (str): _description_
            elem_type (str, optional): _description_. Defaults to 'TextLine'.
            schema (str, optional): _description_. Defaults to ''.
        """
        ocr_dataset = dict()

        ocr_dataset['metainfo'] = {'classes': [elem_type]}
        ocr_dataset['data_list'] = list()

        xmls, imgs = PageTransforms._gather_xmls_imgs(page_path, imgs_path)

        for idx, (image, page) in enumerate(zip(imgs, xmls)):  

            tree = ET.parse(page)
            root = tree.getroot()

            image_instance = dict()
            file_name = Path(image).name
            height, width = PageTransforms._get_img_shape(image)

            schema_formatted = '{' + schema + '}'

            for elem in root.iter(schema_formatted + elem_type):

                image_instance['instances'] = list()
                image_instance['img_path'] = file_name
                image_instance['height'] = height
                image_instance['width'] = width   

                bbox = list()
                poly = list()
                text = ''
                empty_text_field = False

                for child in elem:
                    
                    if child.tag == '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextEquiv':
              
                        for text_field in child:
                    
                            if text_field.text == None or text_field.text == '':
                                empty_text_field = True
                                continue
                            else:
                                text = text_field.text

                    elif child.tag == schema_formatted + 'Coords':
                        bbox, _, poly = PageTransforms._get_bbox_area_poly_from_coords(child)

                if not empty_text_field:
                    image_instance['instances'].append(dict(
                        bbox=bbox,
                        bbox_label=1,
                        mask=poly,
                        text=text
                    ))

                ocr_dataset['data_list'].append(image_instance)

        PageTransforms._write_json(out_path, ocr_dataset)


if __name__ == "__main__":
    pass