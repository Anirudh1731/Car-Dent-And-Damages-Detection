from flask import Flask, request, jsonify, send_file, render_template
import cv2
from ultralytics import YOLO
import os
from config import model,model2,model3,category_colors,category_colors_2

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = file.filename
        image_path = os.path.join('/tmp', filename)
        file.save(image_path)

        # Read the image
        image = cv2.imread(image_path)

        # Make predictions
        results = model.predict(image_path)

        # Initialize variables to keep track of the largest bounding box
        max_area = 0
        largest_box = None

        # Loop through results to find the largest bounding box
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs

            for box in boxes:
                xyxy = box.xyxy[0].tolist()  # Extract coordinates from the list
                x1, y1, x2, y2 = xyxy
                area = (x2 - x1) * (y2 - y1)  # Calculate area

                if area > max_area:
                    max_area = area
                    largest_box = (x1, y1, x2, y2)

        # If a largest bounding box was found, crop the image
        if largest_box is not None:
            x1, y1, x2, y2 = map(int, largest_box)  # Ensure coordinates are integers
            cropped_image = image[y1:y2, x1:x2]  # Crop the image

            cropped_image_path = os.path.join('/tmp', 'largest_cropped_image.jpg')
            cv2.imwrite(cropped_image_path, cropped_image)

            # Use model2 to predict scratches and dents on the cropped image
            cropped_results = model2.predict(cropped_image_path)

            # Draw bounding boxes on the cropped image for detected scratches and dents
            for result in cropped_results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    category_id = box.cls[0].item()  # Get category ID
                    # print(category_id)
                    category_name = category_id  # Get category name
                    color = category_colors.get(category_name, (0, 255, 255))  # Default to yellow if not found
                    cv2.rectangle(cropped_image, (x1, y1), (x2, y2), color, 2)


        

            predicted_image_path = os.path.join('/tmp', 'predicted_cropped_image.jpg')

            cv2.imwrite(predicted_image_path, cropped_image)

            cropped_results_2=model3.predict(predicted_image_path)
            
            for result in cropped_results_2:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    category_id = box.cls[0].item()  # Get category ID
                    category_name = category_id  # Get category name
                    color = category_colors_2.get(category_name, (0, 255, 255))  # Default to yellow if not found
                    print(category_id)
                    cv2.rectangle(cropped_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.rectangle(cropped_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            predicted_image_path_2 = os.path.join('/tmp', 'predicted_cropped_image_2.jpg')

            cv2.imwrite(predicted_image_path_2, cropped_image)

            return jsonify({
                'original_image_url': '/image?filename=' + filename,
                'cropped_image_url': '/image?filename=largest_cropped_image.jpg',
                'predicted_image_url': '/image?filename=predicted_cropped_image.jpg',
                'predicted_image_url_2': '/image?filename=predicted_cropped_image_2.jpg'

            })

        return jsonify({'error': 'No bounding boxes found'}), 400

    return jsonify({'error': 'File upload failed'}), 500

@app.route('/image')
def serve_image():
    filename = request.args.get('filename')
    return send_file(os.path.join('/tmp', filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
