from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("Mold_detector.pt")

# Export the model
model.export(format="openvino")  # creates 'yolo26n_openvino_model/'
