# Hand Gesture Detector (CST-440 Topic 4)

Detects five gestures from a webcam. Every gesture starts from an **open right hand** and moves to an end pose.

## Layout

```
gesture_app/
  frontend/            browser: webcam, hand tracking, coordinate overlay, recording UI
    index.html  style.css  app.js        <- CAMERA CONTROL AREA is at the top of app.js
  backend/             Python: preprocessing, training, prediction API
    config.py  preprocess.py  train_model.py  app.py  requirements.txt
  data/gesture_data.csv        created when you record samples
  models/                      gesture_model.keras, gesture_model_int8.tflite, labels.json
```

**Front end** tracks 21 hand points (MediaPipe Hands) and sends raw coordinates. **Back end** normalises them, trains the model and returns predictions. Normalisation lives only in `backend/preprocess.py`, so training and live use cannot drift apart.

## Gestures

| Class | Start | End pose |
|---|---|---|
| IDLE_OPEN | open right hand | stays open (no gesture) |
| THUMBS_UP | open right hand | fist with thumb pointing up |
| PENCIL | open right hand | thumb and index tips pinched, other three fingers curled (writing grip) |
| ASL_U | open right hand | index and middle fingers up and together, ring and pinky folded, thumb across them |
| ASL_Y | open right hand | thumb and pinky extended, other three fingers curled |
| ASL_F | open right hand | thumb and index tips touching, other three fingers extended |

IDLE_OPEN is a sixth class I added so the model has an answer when no gesture is being made. Remove it from `GESTURES` in `config.py` if you only want five outputs.

## Camera control (64x64 and higher)

At the top of `frontend/app.js`, inside the block marked `CAMERA CONTROL AREA`:

```js
const CAMERA_RES = 64;   // tracking frame is CAMERA_RES x CAMERA_RES
const CAMERA_FPS = 30;
```

Set `CAMERA_RES` to 128, 256 or 480 for a larger frame. Webcams rarely output 64x64 natively, so the code crops the feed to a square and downsizes it to `CAMERA_RES` before tracking; the on-screen view shows that pixelated frame. At 64x64 the hand is only a few pixels wide, so expect jittery or lost tracking. If so, raise the value. Each CSV row stores `camera_res`; record and run at the same value.

## Dataset needed for training

One **sample** is a sequence of **30 frames** (2 seconds at 15 fps) covering open hand → transition → end pose. Each frame holds the 21 hand landmarks, each with x, y, z (63 numbers). Each sample has one label.

CSV columns: `sample_id, user_id, camera_res, frame, label, x0,y0,z0, … x20,y20,z20`

| Field | Type | Meaning |
|---|---|---|
| sample_id | string | groups the 30 rows of one gesture attempt |
| user_id | string | who performed it (split train/test by person if you can) |
| camera_res | int | tracking frame size used (64 by default) |
| frame | int 0–29 | time step within the sample |
| label | string | one of the 6 classes |
| x, y | float 0–1 | landmark position in the frame (fractions of width/height) |
| z | float | depth relative to the wrist (smaller = closer to the camera) |

Illustrative rows (a few of the 63 coordinate columns, values made up for the format):

```
sample_id,label,frame,wrist_x,wrist_y,thumb_tip_x,thumb_tip_y,index_tip_x,index_tip_y,pinky_tip_x,pinky_tip_y
a1f3,THUMBS_UP,0,0.52,0.71,0.38,0.52,0.47,0.30,0.66,0.45
a1f3,THUMBS_UP,8,0.52,0.70,0.40,0.50,0.50,0.40,0.62,0.50
a1f3,THUMBS_UP,15,0.52,0.70,0.45,0.38,0.50,0.55,0.58,0.60
a1f3,THUMBS_UP,29,0.52,0.70,0.46,0.30,0.50,0.58,0.57,0.62
b7c2,ASL_Y,0,0.50,0.72,0.36,0.54,0.46,0.31,0.65,0.44
b7c2,ASL_Y,15,0.50,0.71,0.38,0.52,0.50,0.52,0.67,0.40
b7c2,ASL_Y,29,0.50,0.71,0.37,0.50,0.50,0.56,0.68,0.36
c9d4,ASL_F,0,0.51,0.72,0.37,0.53,0.47,0.30,0.65,0.44
c9d4,ASL_F,15,0.51,0.71,0.44,0.42,0.45,0.42,0.64,0.40
c9d4,ASL_F,29,0.51,0.71,0.45,0.41,0.45,0.41,0.64,0.37
```

How much to collect:
- At least 150 samples per class (about 900 total), more for the look-alike pairs (PENCIL vs ASL_F, THUMBS_UP vs ASL_Y).
- Several people, several distances, lighting conditions and backgrounds, and slightly different hand angles and speeds (the lecture's "same gesture is never performed the same way" problem).
- Include IDLE_OPEN samples where the hand is open and moving around.

## Run it

```bash
cd gesture_app/backend
pip install -r requirements.txt
python app.py                 # open http://localhost:5000, click Start camera
```

1. In the page, pick a gesture, press **Record sample**, and perform open hand → gesture after the countdown. Repeat until you have enough samples for every class.
2. Stop the server, then `python train_model.py`. It splits 70/15/15, trains the Conv1D model from the lecture (input 30x63, six softmax outputs), prints test accuracy and writes the `.keras` and INT8 `.tflite` files.
3. Run `python app.py` again. Live predictions appear beside the camera view.

The page uses a CDN for MediaPipe, so the first load needs internet.
