# Fall Is Coming
By Jérôme, Marie, Mohamad, Thomas

## What ?
Human fall detector with Python and Media Pipe, coded for ACV course.

Can be used as it is, in various contexts such as retirement house, playground. 

Can be integrated in other ACV app such as fitness and yoga trainer, just dance.

## Repo Structure
- `main.py` : to execute the functions
- `ui_show_features` : to detect falls from a video or a webcam 

## Running Guide

```bash
uv sync
uv run streamlit run app.py```

From a webcam live video, detect if someone is falling down.

Conditions to a fall :
- torso is dropping
- torso is going down fast
- the body is lying down

### How to implement 

Requirements to be copied in your repo : 
- file src/extract_landmark.py
- file src/features_extraction.py

1. Init storage class 

    `fe = FeaturesExtraction()`

2. Compute features from frame 

   `fe.compute_feature_from_frame(img_rgb)`

3. State extraction :

    for each frame of your video, those conditions are checked from the features extracted with `fe.frame_to_state()`. If a fall is detected, `fall_state` is True.

    ```python
    #fall_state:bool, height_drop:float, significant_drop_state:bool, angles:float, horizontal_posture_state:bool, fast_downward_state:bool
    fall_state, height_drop, significant_drop_state, angles, horizontal_posture_state, fast_downward_state = fe.frame_to_state()
    ```
    
## Example
You can found an example of webcam live recording displaying the conditions and if a fall is detected in `src/ui_show_features.py` function `ui_features_from_webcam()`.