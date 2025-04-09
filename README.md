<div align="center">

  <h1>Veagle Project</h1>

</div>

## About
Veagle is a tech company providing smart recording cameras for sports facilities, enabling individuals of all levels to track their performance, analyze gameplay, and capture moments through an AI-driven system. The platform also acts as a digital scout, identifying talents and directly connecting them with clubs and academies by sharing their in-game stats and video highlights.

## This AI model ready to:

- **Ball tracking:** Our AI model currently includes ball tracking capabilities. However, due to the ball's small size and its quick movements, particularly in high-resolution footage, this feature is still undergoing refinement to enhance its accuracy.
- **Player tracking:**  The model is equipped to maintain player identification throughout a match. Nonetheless, this process can be challenging due to frequent obstructions from other players or objects on the field, which we are actively working to improve.
- **Player re-identification:** We have integrated player re-identification into our AI system. This function faces complexities, especially when cameras are in motion or when players who look similar leave and re-enter the frame. Our team is diligently training the model to better handle these situations.

## What we promise: 
We are committing to ensuring that by the end of the AI-league, all the following functionalities in our model will be fully developed and operational:
- **Pitch Detection:** The ability to accurately detect and analyze the pitch layout.
- **Player Detection:** Robust detection of players on the field throughout the game.
- **Player tracking:** Reliable detection of the ball, despite its rapid movements and small size.
- **Ball Detection:** Consistent tracking of individual players, managing identification despite occlusions.
- **Team Classification:** Effective classification of players into their respective teams.
- **Radar:** Effective classification of players into their respective teams.


## 💻 Installation: 
```bash
pip install https://github.com/Veagle-Sport/AI.git
cd examples/soccer
pip install -r requirements.txt
```
## Usage: 
```bash
 python main.py --source_video_path data/vid1.mp4 --target_video_path data/vid1-result.mp4 --device cuda --mode PLAYER_TRACKING
 Video to be analyzed: data/vid1.mp4
 Saved Video : data/vid1-result.mp
 Result saved in : AI\examples\soccer\results
```
## datasets

| use case                        | dataset                                                                                                                                                        |
|:--------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Veagle player detection         | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/veagle/veagle/dataset/12)  |
| soccer ball detection           | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)  |
| soccer pitch keypoint detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) |




## Note

This project is modified version of roboflow sports project 
