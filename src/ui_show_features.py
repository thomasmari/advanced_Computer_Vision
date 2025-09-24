from 
video_path = "data/video_chute/istockphoto-1066783428-640_adpp_is.mp4"

def ui_features_from_path():
    #extract all features
    res  = video_to_array(video_path)
    print(res.shape)
    features = cha_table_features(res)
    print(features.shape)
    print(features[:, 0, :])



if __name__ == "__main__":
