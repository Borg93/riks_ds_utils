from glob import glob
from pathlib import Path
import cv2
import os
import xml.etree.ElementTree as ET
import json
import numpy as np
from pathlib import PurePath

class PageTransforms:
    def _gather_xmls_imgs(page_path: str, imgs_path: str):
        print(page_path)

        xmls = glob(os.path.join(page_path, "**"))
        imgs = glob(os.path.join(imgs_path, "**"))
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

    def _get_bbox_area_poly_from_coords(child, width_of_img, height_of_img, x_offset=0, y_offset=0):
        # TODO: change negative values to zero

        width_of_img = int(width_of_img) - 0.5
        height_of_img = int(height_of_img) - 0.5

        coordinates = child.attrib["points"].split()
        temp = [coord.split(",") for coord in coordinates]
        temp = [[int(x) + 0.5 - x_offset, int(y) + 0.5 - y_offset] for [x, y] in temp]
        temp = [(0.5, y) if x <= 0 else (x, y) for (x, y) in temp]
        temp = [(x, 0.5) if y <= 0 else (x, y) for (x, y) in temp]
        temp = [(width_of_img, y) if x >= width_of_img else (x, y) for (x, y) in temp]
        temp = [(x, height_of_img) if y >= height_of_img else (x, y) for (x, y) in temp]
        px = [x for (x, y) in temp]
        py = [y for (x, y) in temp]
        poly = [p for x in temp for p in x]

        try:
            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))
        except Exception as e:
            print(e)
            return (None, None, None)

        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        area = (x_max - x_min) * (y_max - y_min)

        return (bbox, area, poly)

    def _write_json(output_path: str, dataset: dict):
        with open(output_path, "w", encoding="utf8") as f:
            json_str = json.dumps(dataset, indent=4, ensure_ascii=False)
            f.write(json_str)

    def _standardize_eol(ocr_dataset):
        suffix1 = "-"
        suffix2 = "="

        for line in ocr_dataset["data_list"]:
            for annot in line["instances"]:
                annot["text"] = annot["text"].strip()

                if annot["text"].endswith("-"):
                    print(annot["text"])
                    annot["text"] = annot["text"].removesuffix(suffix1) + "¬"
                    print(annot["text"])

                if annot["text"].endswith("="):
                    annot["text"] = annot["text"].removesuffix(suffix2) + "¬"

        return ocr_dataset

    def _get_image_from_page_coords(image: str, element):
        img = cv2.imread(image)

        mask = np.zeros(img.shape[0:2], dtype=np.uint8)

        coordinates = element.attrib["points"].split()
        temp = [coord.split(",") for coord in coordinates]
        temp2 = [[int(x) for x in lst] for lst in temp]
        points = np.array(temp2)
        try:
            cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        except Exception as e:
            print(e)
            return None, None
        res = cv2.bitwise_and(img, img, mask=mask)
        rect = cv2.boundingRect(points)

        wbg = np.ones_like(img, np.uint8) * 255
        cv2.bitwise_not(wbg, wbg, mask=mask)

        # overlap the resulted cropped image on the white background
        dst = wbg + res

        cropped = dst[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

        return rect, cropped

    def page_to_region_coco(xmls: list, imgs: list, out_path: str, elems: list, schema: str = ""):
        """_summary_
        Convert PAGE-files to coco-annotation file for use in training object detection or instance segmentation models
        Put all the PAGE-files in one directory and all the image files in another, they have to be named the same
        except for file-ending.

        Args:
            page_path (str): Path to PAGE files
            imgs_path (str): Path to images
            out_path (str): Path to json-outfile
            elems (list): List of the elements you wish to include in coco, for instance TextLine or TextRegion
            schema (str, optional): xml-schema that the PAGE-files are using, you can look it up in the PAGE-files
        """

        # xmls, imgs = PageTransforms._gather_xmls_imgs(page_path, imgs_path)

        coco_regions = dict()
        coco_regions_annot = list()
        coco_regions_imgs = list()
        coco_regions_categories = list()
        obj_count = 0
        no_of_lines = 0

        for i, el in enumerate(elems):
            coco_regions_categories.append(dict(id=i, name=el))

        for idx, (image, page) in enumerate(zip(imgs, xmls)):
            tree = ET.parse(page)
            root = tree.getroot()

            # filename = Path(image).name
            filename = os.path.sep.join(image.split(os.path.sep)[-3:])
            height, width = PageTransforms._get_img_shape(image)

            coco_regions_imgs.append(dict(id=idx, file_name=filename, height=height, width=width))

            schema_formatted = "{" + schema + "}"

            for i, el in enumerate(elems):
                for elem in root.iter(schema_formatted + el):
                    no_of_lines += 1

                    for child in elem:
                        if child.tag == schema_formatted + "Coords":
                            obj_count += 1

                            bbox, area, poly = PageTransforms._get_bbox_area_poly_from_coords(child, width_of_img=width, height_of_img=height)
                            data_anno = dict(image_id=idx, id=obj_count, category_id=i, bbox=bbox, area=area, segmentation=[poly], iscrowd=0)
                            coco_regions_annot.append(data_anno)

            if idx % 50 == 0:
                print(idx)

        coco_regions["images"] = coco_regions_imgs
        coco_regions["annotations"] = coco_regions_annot
        coco_regions["categories"] = coco_regions_categories

        PageTransforms._write_json(out_path, coco_regions)

    def page_to_mmlabs_ocr(page_path: str, imgs_path: str, out_path: str, elem_type: str = "TextLine", schema: str = ""):
        """_summary_

        Args:
            page_path (str): _description_
            imgs_path (str): _description_
            out_path (str): _description_
            elem_type (str, optional): _description_. Defaults to 'TextLine'.
            schema (str, optional): _description_. Defaults to ''.
        """
        ocr_dataset = dict()

        ocr_dataset["metainfo"] = {"classes": [elem_type]}
        ocr_dataset["data_list"] = list()

        xmls, imgs = PageTransforms._gather_xmls_imgs(page_path, imgs_path)

        for idx, (image, page) in enumerate(zip(imgs, xmls)):
            tree = ET.parse(page)
            root = tree.getroot()

            file_name = Path(image).name
            height, width = PageTransforms._get_img_shape(image)

            schema_formatted = "{" + schema + "}"

            for elem in root.iter(schema_formatted + elem_type):
                image_instance = dict()
                image_instance["instances"] = list()
                image_instance["img_path"] = file_name
                image_instance["height"] = height
                image_instance["width"] = width

                bbox = list()
                poly = list()
                text = ""
                empty_text_field = False

                for child in elem:
                    if child.tag == schema_formatted + "TextEquiv":
                        for text_field in child:
                            if text_field.text == None or text_field.text == "":
                                empty_text_field = True
                                continue
                            else:
                                text = text_field.text

                    elif child.tag == schema_formatted + "Coords":
                        bbox, _, poly = PageTransforms._get_bbox_area_poly_from_coords(child)

                if not empty_text_field and bbox != None:
                    image_instance["instances"].append(dict(bbox=bbox, bbox_label=1, mask=poly, text=text))

                    ocr_dataset["data_list"].append(image_instance)

        ocr_dataset = PageTransforms._standardize_eol(ocr_dataset)

        PageTransforms._write_json(out_path, ocr_dataset)

    def extract_dict_from_page(path_to_page, schema=""):
        xmls = glob(os.path.join(path_to_page, "**"))
        char_set = ""

        for i, xml in enumerate(xmls):
            tree = ET.parse(xml)
            root = tree.getroot()

            schema_formatted = "{" + schema + "}"

            for text_line in root.iter(schema_formatted + "TextLine"):
                for child in text_line:
                    if child.tag == schema_formatted + "TextEquiv":
                        for text_field in child:
                            try:
                                char_list = [char for char in text_field.text]
                                for ch in char_list:
                                    if ch not in char_set:
                                        char_set += ch
                            except Exception as e:
                                print(e)

        char_set_list = [char for char in char_set]
        char_set_list.sort()

        return char_set_list

    def crop_text_reg_write_text_line_coco(page_files: list, imgs: list, coco_out: str):
        # think about folder structure of the resulting dataset

        schema = "{" + "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15" + "}"

        coco_regions = dict()
        coco_lines_annot = list()
        coco_regions_imgs = list()
        obj_count = 0
        img_id = 1

        for idx, (xml, img) in enumerate(zip(page_files, imgs)):
            tree = ET.parse(xml)
            root = tree.getroot()
            region_nr = 0

            for text_region in root.iter(schema + "TextRegion"):
                rect = list()
                cropped = None

                for child in text_region:
                    if child.tag == schema + "Coords":
                        rect, cropped = PageTransforms._get_image_from_page_coords(img, child)

                if cropped is not None:
                    out_folder = os.path.join(*PurePath(img).parts[0:-2], 'text_regions')

                    img_out = os.path.join(out_folder, Path(img).stem + "_" + str(region_nr).zfill(3) + ".jpg")

                    try:
                        cv2.imwrite(img_out, cropped)
                    except:
                        continue

                    region_nr += 1

                    height, width = PageTransforms._get_img_shape(img_out)

                    coco_regions_imgs.append(dict(id=img_id, file_name=os.path.sep.join(img_out.split(os.path.sep)[-3:]), height=height, width=width))

                    for child in text_region:
                        if child.tag == schema + "TextLine":
                            for coords in child:
                                if coords.tag == schema + "Coords":
                                    bbox, area, poly = PageTransforms._get_bbox_area_poly_from_coords(coords, width, height, rect[0], rect[1])
                                    data_anno = dict(
                                        image_id=img_id, id=obj_count, category_id=0, bbox=bbox, area=area, segmentation=[poly], iscrowd=0
                                    )

                                    coco_lines_annot.append(data_anno)
                                    obj_count += 1

                    img_id += 1

            if idx % 50 == 0:
                print(idx)

        coco_regions["images"] = coco_regions_imgs
        coco_regions["annotations"] = coco_lines_annot
        coco_regions["categories"] = [{"id": 0, "name": "text_line"}]

        PageTransforms._write_json(coco_out, coco_regions)

    def crop_line_imgs_page(page_file: str, image: str, dataset_path: str, line_imgs_path: str, gt_path: str, schema: str = ""):
        """_summary_
        Crop line images from page file using field coords(polygon), writes line_images to specified path
        and writes jsonline gt_file to specified path

        Args:
            page_file (str): path to page file
            image (str): path to image
            dataset_path (str): basepath for dataset location
            line_imgs_path (str): outpath for cropped line_images
            gt_path (str): outpath for gt_file
            schema (str, optional): xml_schema
        """

        os.makedirs(line_imgs_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)

        ground_truths = []
        line_number = 0

        # parse xml
        tree = ET.parse(page_file)
        root = tree.getroot()

        schema_formatted = "{" + schema + "}"

        # iterate through text_lines in xml
        for text_line in root.iter(schema_formatted + "TextLine"):
            empty_text_field = True
            coords_extracted = True
            img_name = ""
            gt_line = dict()

            for child in text_line:
                # get ground truth trancription and if not empty, save to gt_list
                if child.tag == schema_formatted + "TextEquiv":
                    for text_field in child:
                        if text_field.text == None or text_field.text == "":
                            continue
                        else:
                            img_name = Path(image).stem + "_" + str(line_number).zfill(4) + ".jpg"
                            gt_line["filename"] = os.path.relpath(os.path.join(line_imgs_path, img_name), start=dataset_path)
                            gt_line["text"] = text_field.text
                            empty_text_field = False

                # get cropping coordinates and crop the text_line, using the polygon mask in the xml-file,
                # mask out everything else, calculate bounding box and crop the line from the masked image
                elif child.tag == "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords":
                    img = cv2.imread(image)
                    mask = np.zeros(img.shape[0:2], dtype=np.uint8)

                    coordinates = child.attrib["points"].split()
                    temp = [coord.split(",") for coord in coordinates]
                    temp2 = [[int(x) for x in lst] for lst in temp]
                    points = np.array(temp2)
                    try:
                        cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
                    except Exception as e:
                        print(e)
                        coords_extracted = False
                        continue
                    res = cv2.bitwise_and(img, img, mask=mask)
                    rect = cv2.boundingRect(points)

                    wbg = np.ones_like(img, np.uint8) * 255
                    cv2.bitwise_not(wbg, wbg, mask=mask)

                    # overlap the resulted cropped image on the white background
                    dst = wbg + res

                    cropped = dst[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
                    # cropped_border = cv2.copyMakeBorder(src=cropped, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255),)

            if not empty_text_field and text_line != "" and bool(gt_line) and coords_extracted:
                try:
                    img_file_path = os.path.join(line_imgs_path, img_name)
                    if not os.path.isfile(img_file_path):
                        cv2.imwrite(img_file_path, cropped)
                    ground_truths.append(gt_line)
                    line_number += 1
                except Exception as e:
                    print(e)
                    continue

        # write gt_file for entire image
        path_to_ground_truths = os.path.join(gt_path, Path(image).stem + "_" + "gt.txt")
        with open(path_to_ground_truths, "w") as f:
            for gt in ground_truths:
                s = json.dumps(gt, ensure_ascii=False)
                f.write(s)
                f.write("\n")


if __name__ == "__main__":
    pass
