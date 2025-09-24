# Feature Extraction for Falls
from extract_landmark import video_to_array
import numpy as np


# def table_features(landmarks):
#     nb_frames, nb_landmarks, nb_coordonnees = np.shape(landmarks)
    
#     nb_landmarks_selectionnes = 7
    
#     features = np.zeros((nb_frames, nb_landmarks_selectionnes, nb_coordonnees * 3 + 2)) # 3 : vecteurs des : coordonnees (0,1,2), vitesse(3,4,5)+6, acce (6,7,8)+9

#     ## COORDONNEES
#     # nez
#     features[:, 0, 0:3] = landmarks[:, 0, :3]

#     # centre épaules
#     features[:, 1, 0:3] = np.mean(landmarks[:, 11:13, :3], axis=1)

#     # centre hanches
#     features[:, 2, 0:3] = np.mean(landmarks[:, 23:25, :3], axis=1)

#     # poignets
#     features[:, 3, 0:3] = landmarks[:, 15, :3]
#     features[:, 4, 0:3] = landmarks[:, 16, :3]

#     # chevilles
#     features[:, 5, 0:3] = landmarks[:, 27, :3]
#     features[:, 6, 0:3] = landmarks[:, 28, :3]

#     ## V & A
#     dt = 1/24 # temps entre 2 frames
#     for idx_landmarks in range(7):
#         # VITESSE 
#         velocity = np.diff(features[:, idx_landmarks, 0:3], axis=0) / dt
#         features[1:, idx_landmarks, 3:6] = velocity
#         features[1:, idx_landmarks, 6] = np.linalg.norm(velocity, axis=1)
#         # ACCELERATION
#         acceleration = np.diff(features[:, idx_landmarks, 4:7], axis=0) / dt
#         features[2:, idx_landmarks, 7:10] = acceleration
#         features[2:, idx_landmarks, 10] = np.linalg.norm(acceleration, axis=1)

#     return features


def cha_table_features(landmarks):
    nb_frames, nb_landmarks, nb_coord = landmarks.shape

    nb_landmarks_selectionnes = 7
    features = np.zeros((nb_frames, nb_landmarks_selectionnes, 11))  # 3 (coord) + 3 (vitesse) + 1 (||v||) + 3 (accel) + 1 (||a||)

    # Landmark indices
    nez = 0
    epaules = [11, 12]
    hanches = [23, 24]
    poignet_gauche = 15
    poignet_droit = 16
    cheville_gauche = 27
    cheville_droite = 28

    # COORDONNÉES
    features[:, 0, 0:3] = landmarks[:, nez, :3]                               # Nez
    features[:, 1, 0:3] = np.mean(landmarks[:, epaules, :3], axis=1)         # Centre épaules
    features[:, 2, 0:3] = np.mean(landmarks[:, hanches, :3], axis=1)         # Centre hanches
    features[:, 3, 0:3] = landmarks[:, poignet_gauche, :3]                   # Poignet gauche
    features[:, 4, 0:3] = landmarks[:, poignet_droit, :3]                    # Poignet droit
    features[:, 5, 0:3] = landmarks[:, cheville_gauche, :3]                  # Cheville gauche
    features[:, 6, 0:3] = landmarks[:, cheville_droite, :3]                  # Cheville droite

    dt = 1 / 24  # Assuming 24 fps

    for idx in range(nb_landmarks_selectionnes):
        # VITESSE vectorielle et scalaire
        velocity = np.diff(features[:, idx, 0:3], axis=0) / dt                # (nb_frames-1, 3)
        features[1:, idx, 3:6] = velocity
        features[1:, idx, 6] = np.linalg.norm(velocity, axis=1)

        # ACCÉLÉRATION vectorielle et scalaire
        acceleration = np.diff(velocity, axis=0) / dt                         # (nb_frames-2, 3)
        features[2:, idx, 7:10] = acceleration
        features[2:, idx, 10] = np.linalg.norm(acceleration, axis=1)

    return features


if __name__ == "__main__":
    video_path = "/home/marie.edet@Digital-Grenoble.local/Documents/mod18_acv/data/chute_ouvrier.mp4"
    res  = video_to_array(video_path)
    print(res.shape)
    features = cha_table_features(res)
    print(features.shape)
    print(features[:, 0, :])