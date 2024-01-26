import os
import torch

import hydra

from boundry_box import draw_boxes, xyxy_to_xywh
from code_tracker import init_tracker
from object_lane import lanes
from ResultsSaver import ResultsSaver
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator

deepsort = init_tracker()


# Class for handling object detection predictions
class DetectionPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_object_counts = None
        self.object_counts = None
        self.txt_path = None
        self.results_data = []  # List to store paths of JSON result files
        self.x = None
        self.lane_counts = {}  # Dictionary to store counts for each lane
        self.total_counts_direction = {}  # Dictionary to store total counts in each direction
        self.results_saver = ResultsSaver(save_dir=self.save_dir)

    def get_annotator(self, img):
        # Create and return an Annotator object with specified parameters
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        # Convert image to PyTorch tensor, apply data type conversion, and normalize pixel values
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        # Apply non-maximum suppression to filter out redundant detections
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
                                        agnostic=self.args.agnostic_nms, max_det=self.args.max_det)

        # Scale and round bounding box coordinates based on the original image shape
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""

        if len(im.shape) == 3:
            pass
        self.seen += 1
        im0 = im0.copy()

        # Determine the frame number based on the source (webcam or dataset)
        if self.webcam:
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        # Reset values for each frame
        self.lane_counts = {}
        self.total_counts_direction = {}
        self.new_object_counts = {}
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        self.annotator = self.get_annotator(im0)

        # Extract predictions for the current frame
        det = preds[idx]
        if len(det) == 0:
            return log_string
        xywh_bboxs = []
        confs = []
        oids = []
        total_counts = 0
        all_object_counts = {"car": 0, "bus": 0, "truck": 0, "cycle": 0}

        # Iterate through each lane and detect objects
        for index, lane in enumerate(lanes):
            counts = 0
            self.object_counts = {object_label: 0 for object_label in ["car", "bus", "truck", "cycle"]}

            # Iterate through each detected bounding box
            for *xyxy, conf, cls in reversed(det):
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                center = (x_c, y_c)

                # Check if the center of the bounding box is inside the current lane
                if lane.is_point_inside(center):
                    counts += 1
                    total_counts += 1

                    self.lane_counts.setdefault(index, 0)
                    self.lane_counts[index] += 1

                    object_label = self.model.names[int(cls)]
                    if object_label in self.object_counts:
                        self.object_counts[object_label] += 1

                    confs.append([conf.item()])
                    oids.append(int(cls))
                    xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)
                    outputs = deepsort.update(xywhs, confss, oids, im0)

                    # Draw bounding boxes with IDs on the image
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -2]
                        object_id = outputs[:, -1]
                        draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)

            log_string += f'Lane{index + 1}:{counts}, '

            # Log object counts for each type of object in the current lane
            for object_label, object_count in self.object_counts.items():
                log_string += f'{object_label}: {object_count}, '

                if object_label in all_object_counts:
                    all_object_counts[object_label] += object_count

            # Check if the current lane is the last one in a group of 3 lanes
            if (index + 1) % 3 == 0:
                direction = ["East", "North", "West", "South"][(index + 1) // 3 - 1]

                # Check if indices are within the valid range
                valid_indices = [i for i in range(index - 2, index + 1) if i in self.lane_counts]

                total_counts_3lanes = sum(self.lane_counts[i] for i in valid_indices)
                json_filepath = self.results_saver.save_results(frame, direction, total_counts_3lanes,
                                                                 self.lane_counts, self.object_counts)

                # Store the path of the saved JSON result file
                self.results_data.append(json_filepath)

                # Update the total counts in the specified direction
                self.total_counts_direction.setdefault(direction, 0)
                self.total_counts_direction[direction] += total_counts_3lanes

        # Print the total number of vehicles in the current frame
        print(f'Frame {frame}: Total number of vehicles - {total_counts}')

        # Save the master results file containing total counts in each direction
        master_json_filepath = self.results_saver.save_master_results(frame, self.total_counts_direction,
                                                                     total_counts, all_object_counts)

        # Store the path of the saved master JSON result file
        self.results_data.append(master_json_filepath)
        return log_string



@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):

    init_tracker()
    cfg.model = cfg.model or "best.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()





if __name__ == "__main__":
    predict()




 # python predict.py model=best7.torchscript source="final.mp4" show=True