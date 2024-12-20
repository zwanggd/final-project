import numpy as np
from scipy.spatial.distance import cosine

class CrossLevelReassignment:
    def __init__(self, iou_threshold=0.5, cosine_similarity_threshold=0.7):
        self.iou_threshold = iou_threshold
        self.cosine_similarity_threshold = cosine_similarity_threshold

    def compute_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def reassign_frame_level(self, frame_detections):
        id_dict = {}
        iou_dict = {}
        for i, bbox1 in enumerate(frame_detections):
            for j, bbox2 in enumerate(frame_detections):
                if i >= j: continue
                iou = self.compute_iou(bbox1['bbox'], bbox2['bbox'])
                if iou > self.iou_threshold:
                    if i in id_dict:
                        old_pair = id_dict[i]
                        if iou > iou_dict[old_pair]:
                            id_dict[i] = j
                            iou_dict[i] = iou
                    elif j in id_dict:
                        old_pair = id_dict[j]
                        if iou > iou_dict[old_pair]:
                            id_dict[j] = i
                            iou_dict[j] = iou
                    else:
                        id_dict[i] = j
                        id_dict[j] = i
                        iou_dict[i] = iou
                        iou_dict[j] = iou
        return id_dict

    def reassign_tracklet_level(self, tracklets):
        reassigned_tracklets = {}
        for i, tracklet1 in enumerate(tracklets):
            for j, tracklet2 in enumerate(tracklets):
                if i >= j: continue
                if len(tracklet1['bboxes']) == 0 or len(tracklet2['bboxes']) == 0: continue
                bbox1 = tracklet1['bboxes'][-1]
                bbox2 = tracklet2['bboxes'][0]
                iou = self.compute_iou(bbox1, bbox2)
                cosine_sim = 1 - cosine(tracklet1['embedding'], tracklet2['embedding'])
                if iou > self.iou_threshold and cosine_sim > self.cosine_similarity_threshold:
                    if i in reassigned_tracklets:
                        old_pair = reassigned_tracklets[i]
                        if iou > self.compute_iou(tracklets[old_pair]['bboxes'][-1], bbox2):
                            reassigned_tracklets[i] = j
                    elif j in reassigned_tracklets:
                        old_pair = reassigned_tracklets[j]
                        if iou > self.compute_iou(bbox1, tracklets[old_pair]['bboxes'][0]):
                            reassigned_tracklets[j] = i
                    else:
                        reassigned_tracklets[i] = j
                        reassigned_tracklets[j] = i
        return reassigned_tracklets

if __name__ == "__main__":
    cross_reassigner = CrossLevelReassignment(iou_threshold=0.5, cosine_similarity_threshold=0.7)
    frame_level_reassignment = cross_reassigner.reassign_frame_level(frame_detections)
    tracklet_level_reassignment = cross_reassigner.reassign_tracklet_level(tracklets)
    
    print("Frame-level reassignment:", frame_level_reassignment)
    print("Tracklet-level reassignment:", tracklet_level_reassignment)
