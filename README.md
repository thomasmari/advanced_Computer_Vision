# Fall Is Coming

## What ?
Human fall detector with Python and Media Pipe, coded for ACV course.

## Files
- `main.py` : to execute the functions
- `ui_show_features` : to detect falls from a video or a webcam 

## How does it work ?
XXX 

From a webcam live video, detect if someone is falling down.

Conditions to a fall :
- torso is dropping
- torso is going down fast
- the body is lying down

For each frame of your video, those conditions are checked from the features extracted with `fe.frame_to_state()`. If a fall is detected, `fall_state` is True.

Example of webcam live recording displaying the conditions and if a fall is detected :
```python


```