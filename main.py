from src.ui_show_features import ui_features_from_webcam, ui_features_from_path

def test_video(path):
    ui_features_from_path(path)

def test_webcam():
    ui_features_from_webcam()


if __name__ == "__main__":
    # test_video(path="/home/marie.edet@Digital-Grenoble.local/Documents/mod18_acv/data/chute_faint.mp4")
    test_webcam()