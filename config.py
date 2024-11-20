from ultralytics import YOLO
model = YOLO('model0.pt')  
model2 = YOLO('model1/weights/best.pt')
model3=YOLO('model2/weights/best.pt')

category_colors = {
    0: (255, 0, 0),  # Red
    1: (0, 255, 0),   # Green
    2: (0, 0, 255)      # Blue
}

category_colors_2 = {
    0: (150, 75, 0),  # brown
    1: (255, 192, 203),   # magenta
    2: (0, 0, 0)      # Black
}

colors={0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255) , 3: (150, 75, 0), 4: (255, 192, 203), 5: (0, 0, 0)}