import cv2
import os
from inference_sdk import InferenceHTTPClient

CONFIDENCE_THRESHOLD = 0.5
THRESHOLD_SMALL = 50
THRESHOLD_MODERATE = 150

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="sHJcxgyCnJTA6cf9uy2w"
)

def classify_pothole(length):
    if length < THRESHOLD_SMALL:
        return "Small"
    elif length < THRESHOLD_MODERATE:
        return "Moderate"
    else:
        return "Large"

def get_pothole_detections(image_path):
    with open(image_path, 'rb') as image_file:
        result = client.infer(image_path, model_id="depthpath/2")

    return result

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    result = get_pothole_detections(image_path)

    for prediction in result['predictions']:
        x, y = int(prediction['x']), int(prediction['y'])
        width, height = int(prediction['width']), int(prediction['height'])
        confidence = prediction['confidence']

        if confidence > CONFIDENCE_THRESHOLD:
            x1, y1 = x - width // 2, y - height // 2
            x2, y2 = x + width // 2, y + height // 2
            size_category = classify_pothole(max(width, height))
            label = f"Pothole: {confidence:.2f}, {size_category}"
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, image)
    return output_path

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = "temp_frame.jpg"
        cv2.imwrite(frame_path, frame)
        result = get_pothole_detections(frame_path)

        for prediction in result['predictions']:
            x, y = int(prediction['x']), int(prediction['y'])
            width, height = int(prediction['width']), int(prediction['height'])
            confidence = prediction['confidence']

            if confidence > CONFIDENCE_THRESHOLD:
                x1, y1 = x - width // 2, y - height // 2
                x2, y2 = x + width // 2, y + height // 2
                size_category = classify_pothole(max(width, height))
                label = f"Pothole: {confidence:.2f}, {size_category}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    return output_path
