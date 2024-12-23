import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


log = logging.getLogger(__name__)


class TrackletTeamClustering(VideoLevelModule):
    """
    This module performs KMeans clustering on the embeddings of the tracklets to cluster the detections with role "player" into two teams.
    Teams are labeled as 0 and 1, and transformer into 'left' and 'right' in a separate module.
    """
    input_columns = ["track_id", "embeddings", "role"]
    output_columns = ["team_cluster"]
    
    def __init__(self, **kwargs):
        super().__init__()
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):

        player_detections = detections[detections.role == "player"]

        embeddings_list = []
        # player_detections.to_csv("/Users/kai/GSR/soccernet/debug2.csv")
        for track_id, group in player_detections.groupby("track_id"):
            if np.isnan(track_id):
                continue
            # numeric_embeddings = [np.array(emb, dtype=np.float32) for emb in group.embeddings.values]
            # embeddings = np.mean(np.vstack(numeric_embeddings), axis=0)
            # log.info(f"group: {print(group.embeddings.values)}")

            embeddings = np.mean(np.vstack(group.embeddings.values), axis=0)
            # valid_embeddings = [e for e in group.embeddings.values if isinstance(e, np.ndarray) and e.size > 0 and np.issubdtype(e.dtype, np.number)]
            # if valid_embeddings:
            #     embeddings_array = np.vstack(valid_embeddings)
            #     embeddings = np.mean(embeddings_array, axis=0)
            # else:
            #     # Dynamically determine the embedding size
            #     desired_embedding_size = valid_embeddings[0].shape[0] if valid_embeddings else 128  # Default to 128 if no embeddings exist
            #     embeddings = np.zeros((desired_embedding_size,))

            embeddings_list.append({'track_id': track_id, 'embeddings': embeddings})

        if not embeddings_list:  # Check if embeddings_list is empty
            detections['team_cluster'] = np.nan  # Initialize 'team_cluster' with a default value
            return detections

        embedding_tracklet = pd.DataFrame(embeddings_list)

        if len(embedding_tracklet) == 1:  # Only one track_id and embedding
            embedding_tracklet['team_cluster'] = 0
        else:
            # Perform KMeans clustering on the embeddings
            embeddings = np.vstack(embedding_tracklet.embeddings.values)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
            embedding_tracklet['team_cluster'] = kmeans.labels_

        # Map the team cluster back to the original detections DataFrame
        detections = detections.merge(embedding_tracklet[['track_id', 'team_cluster']], on='track_id', how='left', sort=False)

        # detections = []
        # log.info(f'tracklet team called')
        return detections
