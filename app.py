from flask import Flask, request, jsonify, render_template_string, Response
import cv2, os, time
import numpy as np
from collections import deque
from ultralytics import YOLO

app = Flask(__name__, static_folder="static")

# ---------------- SETUP ----------------
os.makedirs("static", exist_ok=True)
model = YOLO("best.pt")

# ---------------- DATA ----------------
history = {
    "time": deque(maxlen=20),
    "temp": deque(maxlen=20),
    "spo2": deque(maxlen=20),
    "hr": deque(maxlen=20),
    "bp": deque(maxlen=20),
    "resp": deque(maxlen=20)
}

# ---------------- PREDICTION ----------------
def predict_health(d):
    score = 0
    if d['spo2'] < 94: score += 2
    if d['hr'] > 100: score += 2
    if d['bp'] > 140: score += 2
    if d['resp'] > 22: score += 2
    if d['temp'] > 37.5: score += 1

    if score >= 6: return "🔴 HIGH RISK"
    elif score >= 3: return "🟡 MODERATE RISK"
    else: return "🟢 LOW RISK"

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template_string(TEMPLATE)

@app.route("/submit", methods=["POST"])
def submit():
    data = request.json
    now = time.strftime("%H:%M:%S")

    for k in history:
        if k != "time":
            history[k].append(float(data[k]))
    history["time"].append(now)

    prediction = predict_health(data)

    return jsonify({
        "history": {k: list(v) for k, v in history.items()},
        "prediction": prediction
    })

# -------- IMAGE DETECTION --------
@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files["file"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    plotted = results[0].plot()
    path = "static/image_result.jpg"
    cv2.imwrite(path, plotted)
    return jsonify({
        "image": "/static/image_result.jpg?t=" + str(time.time())
    })

# -------- VIDEO UPLOAD --------
@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files["file"]
    file.save("temp_input.mp4")
    return jsonify({"status": "uploaded"})

# -------- STREAM VIDEO --------
@app.route("/stream_video")
def stream_video():
    def generate():
        cap = cv2.VideoCapture("temp_input.mp4")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res = model(frame)
            frame = res[0].plot()
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- UI ----------------
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body { margin:0; font-family:'Segoe UI'; background:linear-gradient(135deg,#f8f7ff,#ede9fe); }
.header { background:linear-gradient(90deg,#6d28d9,#9333ea); color:white; padding:18px; text-align:center; font-size:26px; }
.footer { background:#6d28d9; color:white; text-align:center; padding:10px; position:fixed; bottom:0; width:100%; }
.container { padding:25px; max-width:1100px; margin:auto; }
.tabs button { background:#8b5cf6; color:white; border:none; padding:10px 18px; border-radius:10px; margin-right:10px; }
.card { background:white; padding:20px; border-radius:14px; margin-bottom:20px; box-shadow:0 6px 25px rgba(0,0,0,0.08); }
input { padding:10px; margin:6px; border-radius:8px; border:1px solid #ccc; }
button.action { background:#7c3aed; color:white; padding:10px 16px; border:none; border-radius:10px; cursor:pointer; }
.preview { margin-top:12px; max-width:100%; border-radius:12px; }
</style>
</head>
<body>
<div class="header">Real-Time Chest Pain & Fainting Detection and Emergency Alert System</div>
<div class="container">
<div class="tabs">
<button onclick="show('health')">Health</button>
<button onclick="show('fall')">Fall Detection</button>
</div>

<!-- HEALTH -->
<div id="health">
<div class="card">
<input id="temp" placeholder="Temperature">
<input id="spo2" placeholder="SpO2">
<input id="hr" placeholder="Heart Rate">
<input id="bp" placeholder="BP">
<input id="resp" placeholder="Resp Rate">
<br>
<button class="action" onclick="predict()">Predict</button>
<button class="action" onclick="simulate()">Simulate</button>
<div id="pred"></div>
</div>
<div class="card">
<canvas id="chart"></canvas>
</div>
</div>

<!-- FALL -->
<div id="fall" style="display:none">
<div class="card">
<h3>📷 Image Detection</h3>
<input type="file" id="img">
<button class="action" onclick="uploadImage()">Detect</button>
<img id="imgOut" class="preview">
</div>
<div class="card">
<h3>🎥 Video Detection (Streaming)</h3>
<input type="file" id="vid">
<button class="action" onclick="startStream()">Detect Fall</button>
<img id="stream" class="preview">
</div>
</div>
</div>

<div class="footer">© 2026 Smart Healthcare AI System</div>

<script>
let chart;
function show(id){
 document.getElementById('health').style.display='none';
 document.getElementById('fall').style.display='none';
 document.getElementById(id).style.display='block';
}
async function predict(){
 let data={ temp:+temp.value, spo2:+spo2.value, hr:+hr.value, bp:+bp.value, resp:+resp.value };
 let res=await fetch('/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
 let d=await res.json();
 pred.innerText="Prediction: "+d.prediction;
 if(chart) chart.destroy();
 chart=new Chart(document.getElementById('chart'),{
  type:'line',
  data:{
   labels:d.history.time,
   datasets:[
    {label:'Temp',data:d.history.temp},
    {label:'SpO2',data:d.history.spo2},
    {label:'HR',data:d.history.hr},
    {label:'BP',data:d.history.bp},
    {label:'Resp',data:d.history.resp}
   ]
  }
 });
}
function simulate(){
 temp.value=(36+Math.random()*2).toFixed(1);
 spo2.value=(90+Math.random()*10);
 hr.value=(60+Math.random()*60);
 bp.value=(100+Math.random()*60);
 resp.value=(12+Math.random()*10);
 predict();
}
async function uploadImage(){
 let f=new FormData();
 f.append('file',img.files[0]);
 let r=await fetch('/upload_image',{method:'POST',body:f});
 let d=await r.json();
 imgOut.src=d.image;
}
async function startStream(){
 let f=new FormData();
 f.append('file',vid.files[0]);
 await fetch('/upload_video',{method:'POST',body:f});
 document.getElementById('stream').src="/stream_video";
}
</script>
</body>
</html>
"""

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
