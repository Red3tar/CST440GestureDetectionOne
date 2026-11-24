/* =====================================================================
 *  FRONT END: camera + hand tracking + display.
 *  The back end (Flask) owns preprocessing, training and prediction.
 * ===================================================================== */

/* ##################################################################### */
/* #                      CAMERA CONTROL AREA                          # */
/* #  Change CAMERA_RES to raise or lower the frame size.              # */
/* #  64 = default (64x64).  Try 128, 256 or 480 for better tracking.  # */
/* #  Record training data and run live detection at the SAME value.   # */
/* ##################################################################### */
const CAMERA_RES = 64;      // width = height of the frame used for hand tracking
const CAMERA_FPS = 30;      // frame rate requested from the webcam
/* ##################################################################### */
/* #                    END CAMERA CONTROL AREA                        # */
/* ##################################################################### */

const API = "";             // same origin as the Flask back end
const DISPLAY = 512;        // on-screen canvas size (the tracked frame is scaled up to this)
const PREDICT_EVERY_MS = 500;

// Filled from /api/config so front end and back end always agree
let LABELS = [];
let SEQ_LEN = 30;
let SAMPLE_FPS = 15;

const video = document.createElement("video");
video.playsInline = true;
video.muted = true;

// Frame at CAMERA_RES x CAMERA_RES that MediaPipe actually sees
const capture = document.createElement("canvas");
capture.width = CAMERA_RES;
capture.height = CAMERA_RES;
const capCtx = capture.getContext("2d", { willReadFrequently: true });

const view = document.getElementById("view");
const ctx = view.getContext("2d");
const el = (id) => document.getElementById(id);

// Landmark names shown in the coordinate table (MediaPipe indices)
const TRACKED = { Wrist: 0, "Thumb tip": 4, "Index tip": 8, "Middle tip": 12, "Ring tip": 16, "Pinky tip": 20 };

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

let buffer = [];            // last SEQ_LEN sampled frames (each: 21 x [x, y, z])
let lastSample = 0;
let lastSeen = 0;
let lastPredict = 0;
let recording = null;       // {frames: [], label} while recording
let running = false;

/* ------------------------------------------------------------ camera */
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: CAMERA_RES },
      height: { ideal: CAMERA_RES },
      frameRate: { ideal: CAMERA_FPS },
      facingMode: "user",
    },
  });
  video.srcObject = stream;
  await video.play();
  const s = stream.getVideoTracks()[0].getSettings();
  // Most webcams cannot output 64x64, so we crop to a square and downscale in drawCapture().
  el("camInfo").textContent =
    `Webcam delivers ${s.width}x${s.height}. Tracking frame: ${CAMERA_RES}x${CAMERA_RES} at ${CAMERA_FPS} fps.`;
}

function drawCapture() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const side = Math.min(vw, vh);
  capCtx.imageSmoothingEnabled = true;
  capCtx.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side, 0, 0, CAMERA_RES, CAMERA_RES);
}

/* ------------------------------------------------------------ hand tracking */
const hands = new Hands({
  locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${f}`,
});
hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
hands.onResults(onResults);

function onResults(results) {
  // Draw the (low-res) frame the model sees, scaled up without smoothing
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(capture, 0, 0, DISPLAY, DISPLAY);

  const lm = results.multiHandLandmarks && results.multiHandLandmarks[0];
  if (!lm) {
    if (performance.now() - lastSeen > 500) { buffer = []; }
    el("confidence").textContent = "Show your right hand to the camera.";
    return;
  }
  lastSeen = performance.now();
  drawHand(lm);
  showCoords(lm);

  // Sample at a fixed rate so sequences match the training data regardless of webcam fps
  const now = performance.now();
  if (now - lastSample >= 1000 / SAMPLE_FPS) {
    lastSample = now;
    const frame = lm.map((p) => [p.x, p.y, p.z]);   // raw, NOT mirrored, NOT normalised
    buffer.push(frame);
    if (buffer.length > SEQ_LEN) buffer.shift();
    if (recording) collect(frame);
  }

  if (!recording && buffer.length === SEQ_LEN && now - lastPredict > PREDICT_EVERY_MS) {
    lastPredict = now;
    predict([...buffer]);
  }
}

function drawHand(lm) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#35c2f0";
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(lm[a].x * DISPLAY, lm[a].y * DISPLAY);
    ctx.lineTo(lm[b].x * DISPLAY, lm[b].y * DISPLAY);
    ctx.stroke();
  }
  ctx.fillStyle = "#ffd166";
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(p.x * DISPLAY, p.y * DISPLAY, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function showCoords(lm) {
  el("coordRows").innerHTML = Object.entries(TRACKED).map(([name, i]) =>
    `<tr><td>${name}</td><td>${lm[i].x.toFixed(3)}</td><td>${lm[i].y.toFixed(3)}</td><td>${lm[i].z.toFixed(3)}</td></tr>`
  ).join("");
}

/* ------------------------------------------------------------ calls to the back end */
async function predict(frames) {
  try {
    const r = await fetch(`${API}/api/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ frames }),
    });
    const data = await r.json();
    if (!r.ok) { el("confidence").textContent = data.error; return; }
    el("result").textContent = data.label.replace("_", " ");
    el("confidence").textContent = `Confidence ${(data.confidence * 100).toFixed(0)}%`;
    renderBars(data.probabilities);
  } catch (e) {
    el("confidence").textContent = "Cannot reach the back end. Is app.py running?";
  }
}

function renderBars(probs) {
  el("bars").innerHTML = Object.entries(probs).map(([name, p]) =>
    `<li><span>${name}</span><span class="track"><span class="fill" style="width:${(p * 100).toFixed(0)}%;display:block"></span></span><span>${(p * 100).toFixed(0)}%</span></li>`
  ).join("");
}

/* ------------------------------------------------------------ recording training samples */
async function startRecording() {
  const status = el("recordStatus");
  el("recordBtn").disabled = true;
  for (let n = 3; n > 0; n--) { status.textContent = `Open your right hand. Starting in ${n}…`; await sleep(1000); }
  status.innerHTML = `<span class="recording">Recording:</span> move from open hand to ${el("labelSelect").value.replace("_", " ")}`;
  recording = { frames: [], label: el("labelSelect").value };
}

function collect(frame) {
  recording.frames.push(frame);
  if (recording.frames.length < SEQ_LEN) return;
  const done = recording;
  recording = null;
  saveSample(done);
}

async function saveSample({ frames, label }) {
  try {
    const r = await fetch(`${API}/api/sample`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames, label, user_id: el("userId").value || "anon", camera_res: CAMERA_RES }),
    });
    const data = await r.json();
    el("recordStatus").textContent = r.ok ? `Saved ${label} (${data.frames} frames).` : `Not saved: ${data.error}`;
  } catch (e) {
    el("recordStatus").textContent = "Not saved: cannot reach the back end.";
  }
  el("recordBtn").disabled = false;
}

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

/* ------------------------------------------------------------ main loop */
async function loop() {
  if (!running) return;
  drawCapture();
  await hands.send({ image: capture });
  requestAnimationFrame(loop);
}

el("startBtn").addEventListener("click", async () => {
  try {
    const cfg = await (await fetch(`${API}/api/config`)).json();
    LABELS = cfg.labels; SEQ_LEN = cfg.seq_len; SAMPLE_FPS = cfg.sample_fps;
    el("labelSelect").innerHTML = LABELS.map((l) => `<option>${l}</option>`).join("");
    await startCamera();
  } catch (e) {
    el("camInfo").textContent = `Could not start: ${e.message}. Check camera permission and that app.py is running.`;
    return;
  }
  el("startBtn").hidden = true;
  el("recordBtn").disabled = false;
  running = true;
  loop();
});

el("recordBtn").addEventListener("click", startRecording);
