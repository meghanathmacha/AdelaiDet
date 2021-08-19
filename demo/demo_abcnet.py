import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import pkg_resources
import json

from symspellpy import SymSpell, Verbosity
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

import numpy as np
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import torch
import matplotlib.pyplot as plt

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from adet.utils.visualizer import TextVisualizer

MAX_EDIT_DISTANCE = 6
DICTIONARY_PATH = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")


class ABCNETDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, post_process=False, dictionary = None):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            post_process(bool) : Should post-processing be performed after de-coding?
            dictionary(SymSpell) : Dictionary object to perform the post-processing.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = cfg.MODEL.ROI_HEADS.NAME == "TextHead"

        self.predictor = DefaultPredictor(cfg)

        self.post_process = post_process
        self.dictionary = dictionary
        if self.post_process:
            assert self.dictionary != None, "Please provide a dictionary for post-processing."

    def _decode_recognition(self, rec):
        CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

        s = ''
        for c in rec:
            c = int(c)
            if c < 95:
                s += CTLABELS[c]
            elif c == 95:
                s += u'口'
        return s

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            text_predictions = self.draw_instance_predictions(predictions=instances)

        return predictions, text_predictions

    def draw_instance_predictions(self, predictions):
        beziers = predictions.beziers.numpy()
        scores = predictions.scores.tolist()
        recs = predictions.recs

        all_texts = []
        all_scores =[]
        for bezier, rec, score in zip(beziers, recs, scores):
            text = self._decode_recognition(rec)
            if self.post_process and text.isalpha():
                res_texts = self.dictionary.lookup(text, Verbosity.CLOSEST, max_edit_distance=4, transfer_casing = True, include_unknown = True, ignore_token="[0-9%.&$@#!(),]")
                text = res_texts[0].term

            all_texts.append(text)
            all_scores.append(score)

        return dict(zip(all_texts, all_scores))


def setup_cfg(args):

    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--post_process",
        help="Post process the recognized text with a English dictionary",
        default = False,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # Load the English dictionary for post-processing
    if args.post_process:
        sym_spell = SymSpell(max_dictionary_edit_distance = MAX_EDIT_DISTANCE)
        sym_spell.load_dictionary(DICTIONARY_PATH, 0 , 1)
        logger.info("Dictionary loaded successfully for post-processing.")
        demo = ABCNETDemo(cfg, post_process=args.post_process, dictionary = sym_spell)
    else:
        demo = ABCNETDemo(cfg)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, text_predictions = demo.run_on_image(img)
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(text_predictions.keys()), time.time() - start_time
                )
            )

            #time.sleep(1)

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    #out_filename = os.path.join(args.output, os.path.basename(path))
                    json_filename = os.path.join(args.output, os.path.basename(path).split(".")[0] + ".txt")
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"

                with open(json_filename, 'w') as f:
                    json.dump(text_predictions, f, indent = 4)

                logger.info(
                "{}: has the json output".format(
                    json_filename
                    )
                )
            else:
                assert args.output , "Please specify a directory with args.output"
