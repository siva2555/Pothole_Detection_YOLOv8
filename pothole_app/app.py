from flask import Flask, render_template, request, send_from_directory
import os
from pothole_detection import process_image, process_video

app = Flask(__name__)
UPLOAD_FOLDER = 'static/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return "No file selected", 400

    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    ext = filename.split('.')[-1].lower()
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{filename}")

    if ext in ['jpg', 'jpeg', 'png']:
        result_path = process_image(file_path, output_path)
    elif ext in ['mp4', 'avi', 'mov']:
        result_path = process_video(file_path, output_path)
    else:
        return "Unsupported file type", 400

    return {"output_path": result_path.replace("static/", "")}

@app.route('/static/output/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
