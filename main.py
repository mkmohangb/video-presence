import cv2
import os
import json
import numpy as np
import time
from collections import defaultdict

from surya.input.load import load_from_file
from surya.model.detection.model import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
import sys

def main():
    folder_name = "output"

    det_processor = load_detection_processor()
    det_model = load_detection_model()

    rec_model = load_recognition_model()
    rec_processor = load_recognition_processor()

    result_path = folder_name
    os.makedirs(result_path, exist_ok=True)
    impath = sys.argv[1]

    images, names, _ = load_from_file(impath)
    image_langs = [['en']]

    start = time.time()
    predictions_by_image = run_ocr(images, image_langs, det_model, det_processor, rec_model, rec_processor, highres_images=None)
    if True:
        print(f"OCR took {time.time() - start:.2f} seconds")
        max_chars = max([len(l.text) for p in predictions_by_image for l in p.text_lines])
        print(f"Max chars: {max_chars}")

        if True:
            for idx, (name, image, pred, langs) in enumerate(zip(names, images, predictions_by_image, image_langs)):
                bboxes = [l.bbox for l in pred.text_lines]
                pred_text = [l.text for l in pred.text_lines]
                page_image = draw_text_on_image(bboxes, pred_text, image.size, langs, has_math="_math" in langs if langs else False)
                page_image.save(os.path.join(result_path, f"{name}_{idx}_text.png"))

        out_preds = defaultdict(list)
        for name, pred, image in zip(names, predictions_by_image, images):
            out_pred = pred.model_dump()
            out_pred["page"] = len(out_preds[name]) + 1
            out_preds[name].append(out_pred)

        print(out_preds)
        #with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        #    json.dump(out_preds, f, ensure_ascii=False)
        def is_same_line(loc1, loc2):
            return abs(loc1[0] - loc2[0]) <= 5 and abs(loc1[1] - loc2[1]) <= 5

        cur_loc = None
        line_count = 0
        attendance = {}
        cur_name = ''
        video_status = None

        img = cv2.imread(impath, 0)
        output = out_preds[names[0]][0]
        for line in output['text_lines']:
            loc = (line['bbox'][1], line['bbox'][3])
            if cur_loc == None:
                cur_loc = loc
                line_count += 1
            if not is_same_line(loc, cur_loc):
                attendance[cur_name] = video_status
                cur_name = line['text']
                cur_loc = loc
                line_count += 1
                bbox = line['bbox']
                if line_count > 2:
                    roi = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):img.shape[1]]
                    ext = roi[0:roi.shape[0], roi.shape[1] - 60: roi.shape[1]]
                    cv2.imwrite(f"output/{cur_name}.jpg", ext)
                    video_status = 'off' if detect_diagonal_line(ext) else 'on'
            elif len(line['text']) > len(cur_name):
                cur_name = line['text']
        attendance[cur_name] = video_status

        print("\n\n\n")
        import pprint
        pprint.pp(attendance)

def detect_diagonal_line(img):
    """
    Detect the presence of a diagonal line using contours and line fitting.
    
    Parameters:
    image_path (str): Path to the input image
    
    Returns:
    bool: True if diagonal line is detected, False otherwise
    """
    
    # Convert to binary - use a simple threshold since it's a clear black line
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    is_diagonal = False
    
    for contour in contours:
        
        # Fit a line to the contour points
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calculate angle in degrees
        angle = np.degrees(np.arctan2(vy, vx))
        # Normalize angle to 0-180
        if angle < 0:
            angle += 180
            
        # Check if the line is diagonal (around 45 or 135 degrees)
        is_diagonal = (abs(angle - 135) <= 15)
        if is_diagonal:
            break
    
    return is_diagonal

if __name__ == "__main__":
    main()

