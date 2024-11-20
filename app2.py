from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import os
from config import model, model2, model3, category_colors, category_colors_2, colors

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def checkLargest(results):
    for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs

            for box in boxes:
                xyxy = box.xyxy[0].tolist()  # Extract coordinates from the list
                x1, y1, x2, y2 = xyxy
                area = (x2 - x1) * (y2 - y1)  # Calculate area

                if area > max_area:
                    max_area = area
                    largest_box = (x1, y1, x2, y2)
    return largest_box

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the image from the uploaded file
        image = np.array(Image.open(file.stream))

        # Make predictions with the first model
        results = model.predict(image)

        # Initialize variables to keep track of the largest bounding box
        max_area = 0
        largest_box = None

        largest_box=checkLargest(results)

        # Loop through results to find the largest bounding box
        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs

        #     for box in boxes:
        #         xyxy = box.xyxy[0].tolist()  # Extract coordinates from the list
        #         x1, y1, x2, y2 = xyxy
        #         area = (x2 - x1) * (y2 - y1)  # Calculate area

        #         if area > max_area:
        #             max_area = area
        #             largest_box = (x1, y1, x2, y2)

        # If a largest bounding box was found, crop the image
        if largest_box is not None:
            x1, y1, x2, y2 = map(int, largest_box)  # Ensure coordinates are integers
            cropped_image = image[y1:y2, x1:x2]  # Crop the image

            # Use model2 to predict scratches and dents on the cropped image
            cropped_results = model2.predict(cropped_image)

            # Combine predictions from both models and draw bounding boxes
            combined_results = []

            # Offset for the second model's category IDs
            offset = 3
            res = model2.names
            res2 = {k + offset: v for k, v in model3.names.items()}
            res.update(res2)

            for result in cropped_results:
                for box in result.boxes:
                    combined_results.append((box, (0, 255, 255), box.cls[0].item()))

            cropped_results_2 = model3.predict(cropped_image)

            for result in cropped_results_2:
                for box in result.boxes:
                    combined_results.append((box, (0, 255, 0), box.cls[0].item() + offset))

            # Prepare JSON response data
            json_results = []
            for box, default_color, cat_id in combined_results:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                category_id = cat_id  # Get category ID
                confidence = box.conf[0].item()  # Get confidence score
                category_name = res.get(category_id)
                color = colors.get(category_id, (0, 255, 0))  # Use appropriate color

                result_dict = {
                    "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "confidence": confidence,
                    "color": color,
                    "category_id": category_id,
                    "category_name": category_name
                }
                json_results.append(result_dict)

                # Draw bounding boxes on the cropped image for combined results
                cv2.rectangle(cropped_image, (x1, y1), (x2, y2), color, 2)
                label = f'{category_name}: {confidence:.2f}'
                cv2.putText(cropped_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert cropped_image to a format suitable for sending as a response
            # _, img_encoded = cv2.imencode('.jpg', cropped_image)
            # img_bytes = BytesIO(img_encoded)

            return jsonify(json_results)

        return jsonify({'error': 'No bounding boxes found'}), 400

    return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
