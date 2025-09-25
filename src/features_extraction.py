import numpy as np
from src.extract_landmark import video_to_array, frame_to_row

class FeaturesExtraction():
    def __init__(self,size=50):
        nb_landmarks_selectionnes = 7
        nb_features = 11 # x,y,z,x',y',z',||v||, x'',y'',z'',||a||
        self.size = size
        self.features = np.zeros((size, nb_landmarks_selectionnes, nb_features))  # 3 (coord) + 3 (vitesse) + 1 (||v||) + 3 (accel) + 1 (||a||)
        self.last_id = -1
        self.frame_number = 0
        self.last_FD_condition = -1000
        self.last_SD_condition = -1000
        self.last_HP_condition = -1000
        self.last_FALL_condition = -1000

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

    def compute_feature_from_frame(self, img_rgb):
        """ Extract frame information
        Store in the storage class the information of the new frame
        Input:
        * imf_rgb an rgb frame
        """
        self.compute_feature_row(frame_to_row(img_rgb))

    def get_previous_feature_row(self, index=0):
        """ return internal format of storage 
        use get position, velocity or acceleration
        index = 0, for the last frame"""
        if index >= self.size:
            return Exception("Cannot access rows with index greater that init size of instance")
        return self.features[(self.last_id - index)]
    
    def get_previous_feature_rows(self, number=1):
        """index = 0, for the last frame"""
        if (self.last_id-number)>= 0:
            return(self.features[self.last_id-number:self.last_id+1].copy())
        else:
            return (np.concatenate(self.features[self.last_id-number:], self.features[:self.last_id+1]))

    def get_shoulder_vy(self):
        return(self.get_previous_feature_row()[1,4])
    
    def get_shoulder_ay(self):
        return(self.get_previous_feature_row()[1,8])
    
    def get_hip_vy(self):
        return(self.get_previous_feature_row()[2,4])
    
    def get_hip_ay(self):
        return(self.get_previous_feature_row()[2,8])
    
    def get_fast_downward_state(self,velocity_threshold=0.5, acceleration_threshold=5.0):
        fast_downward = (self.get_shoulder_vy() > velocity_threshold and
                         self.get_hip_vy() > velocity_threshold and
                         self.get_shoulder_ay() > acceleration_threshold and
                         self.get_hip_ay() > acceleration_threshold)
        if fast_downward:
            self.last_FD_condition = self.frame_number
        return(fast_downward)
    
    def compute_torso_angle(self, shoulders, hips):
        """
        Returns angle (in degrees) between torso vector (shoulders to hips) and vertical axis (y-axis).
        0° = upright, 90° = lying down
        """
        torso_vector = hips - shoulders  # shape: (nb_frames, 3)
        vertical_vector = np.array([0, -1, 0])  # y-axis up
        dot_product = np.sum(torso_vector * vertical_vector)
        norms = np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector)
        cos_theta = np.clip(dot_product / norms, -1.0, 1.0)

        angles = np.arccos(cos_theta) * 180 / np.pi  # convert to degrees
        return angles  # shape: (nb_frames,)

    def get_angles(self):
        # Reconstruct positions to compute orientation
        shoulders = self.get_previous_feature_row()[1,0:3]
        hips = self.get_previous_feature_row()[2,0:3]
        angles = np.round(self.compute_torso_angle(shoulders, hips),2)  # 0° = upright, 90° = flat
        return(angles)
    
    def get_horizontal_posture_state(self, angle_threshold=100.0):
        bool_HP_state = self.get_angles() < angle_threshold
        if bool_HP_state:
            self.last_HP_condition = self.frame_number
        return(bool_HP_state)

    def get_height_drop(self):
        features = self.get_previous_feature_row()
        features_20 = self.get_previous_feature_row(20)
        initial_torso_y = (features_20[1,1] + features_20[2,1]) / 2
        current_torso_y = (features[1,1] + features[2,1]) / 2
        height_drop = current_torso_y - initial_torso_y  # y increases downward in MediaPipe
        return(np.round(height_drop, 3))

    def get_significant_drop_state(self, height_drop_threshold=0.2):
        significant_drop = self.get_height_drop() > height_drop_threshold
        if significant_drop:
            self.last_SD_condition = self.frame_number
        return(significant_drop)
    
    def get_fall_state(self, fall_cool_down = 50, analysis_window = 20):
        if self.frame_number - self.last_FALL_condition < fall_cool_down:   
            return True
        if (self.frame_number-self.last_HP_condition < analysis_window) and (self.frame_number-self.last_SD_condition < analysis_window) and (self.frame_number-self.last_FD_condition < analysis_window) :
        # if self.get_horizontal_posture_state() or self.get_significant_drop_state() or self.get_fast_downward_state():
            self.last_FALL_condition = self.frame_number
            return True
        return False
    
    def frame_to_state(self):
        """ process frame to extract fall status
        return:
        * fall_state : Main fall detection (True if the 3 falls states are True), False

        * height_drop : float, how much the torso dropped
        * significant_drop_state : True if the dorso drop is considered significant
        * angles : float, angle of the body to horizontal
        * horizontal_posture_state : True if the body is horizontal, False
        * fast_downward_state : True if torso dropped fast enough to be considered a fall
        """
        fall_state = self.get_fall_state()
        height_drop = self.get_height_drop()
        if self.frame_number >= 20:
            significant_drop_state = self.get_significant_drop_state()
        else:
            significant_drop_state = False
        angles = self.get_angles()
        horizontal_posture_state = self.get_horizontal_posture_state()
        fast_downward_state = self.get_fast_downward_state()
        return fall_state, height_drop, significant_drop_state, angles, horizontal_posture_state, fast_downward_state
    
    def get_position(self,index=0):
        frame=self.get_previous_feature_row(index)
        return frame[:,0:3]
    
    def get_velocity(self,index=0):
        frame=self.get_previous_feature_row(index)
        return frame[:,3:6]
    
    def get_acceletation(self,index=0):
        frame=self.get_previous_feature_row(index)
        return frame[:,7:10]
    
