import numpy as np

class FeaturesExtraction():
    def __init__(self,size=50):
        nb_landmarks_selectionnes = 7
        nb_features = 11 # x,y,z,x',y',z',||v||, x'',y'',z'',||a||
        self.size = size
        self.features = np.zeros((size, nb_landmarks_selectionnes, nb_features))  # 3 (coord) + 3 (vitesse) + 1 (||v||) + 3 (accel) + 1 (||a||)
        self.last_id = -1
        self.frame_number = 0

    def compute_feature_row(self,landmark_row):
        #class attribute
        self.last_id = (self.last_id + 1) % self.size
        #functional core 
        nb_landmarks, nb_coord = landmark_row.shape
        nb_landmarks_selectionnes = 7

        # Landmark indices
        nez = 0
        epaules = [11, 12]
        hanches = [23, 24]
        poignet_gauche = 15
        poignet_droit = 16
        cheville_gauche = 27
        cheville_droite = 28

        # COORDONNÉES
        self.features[self.last_id, 0, 0:3] = landmark_row[nez, :3]                               # Nez
        self.features[self.last_id, 1, 0:3] = np.mean(landmark_row[epaules, :3], axis=0)         # Centre épaules
        self.features[self.last_id, 2, 0:3] = np.mean(landmark_row[hanches, :3], axis=0)         # Centre hanches
        self.features[self.last_id, 3, 0:3] = landmark_row[poignet_gauche, :3]                   # Poignet gauche
        self.features[self.last_id, 4, 0:3] = landmark_row[poignet_droit, :3]                    # Poignet droit
        self.features[self.last_id, 5, 0:3] = landmark_row[cheville_gauche, :3]                  # Cheville gauche
        self.features[self.last_id, 6, 0:3] = landmark_row[cheville_droite, :3]                  # Cheville droite

        dt = 1 / 24  # Assuming 24 fps
        # VITESSE
        if self.frame_number>0:
            for idx in range(nb_landmarks_selectionnes):
                # VITESSE vectorielle et scalaire
                velocity = (self.features[self.last_id, idx, 0:3] - self.get_previous_feature_row(1)[idx, 0:3])/ dt 
                self.features[self.last_id, idx, 3:6] = velocity
                self.features[self.last_id, idx, 6] = np.linalg.norm(velocity)
        if self.frame_number>1:
            for idx in range(nb_landmarks_selectionnes):
                # ACCÉLÉRATION vectorielle et scalaire
                acceleration = (self.features[self.last_id, idx, 3:6] - self.get_previous_feature_row(1)[idx, 3:6])/ dt                        # (nb_frames-2, 3)
                self.features[2:, idx, 7:10] = acceleration
                self.features[2:, idx, 10] = np.linalg.norm(acceleration)
        #class atribute
        self.frame_number += 1

        

    def get_previous_feature_row(self, index=0):
        """index = 0, for the last frame"""
        if index >= self.size:
            return Exception("Cannot access rows with index greater that init size of instance")
        return self.features[(self.last_id - index)]
