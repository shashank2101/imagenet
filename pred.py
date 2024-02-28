import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from mtcnn import MTCNN
IMAGE_SIZE=(224,224)
BATCH_SIZE=32
# Assuming IMAGE_SIZE and train_generator are defined in exp.py
# from exp import IMAGE_SIZE, train_generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split 20% of data for validation
)
data_dir = 'D:/attendance/chrome/database1'

# Generate data batches from directory
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # Use subset for training data
)
detector = MTCNN()
model = load_model('fixed.h5')
# Load the image
img_path = 'D:/attendance/chrome/ko2.jpeg'  # Replace with the path to your single image
img = cv2.imread(img_path)

# Detect faces in the image
faces = detector.detect_faces(img)

# If faces are detected, process the first face found
if faces:
    # Extract the first face found
    face_data = faces[0]
    x, y, w, h = face_data['box']
    
    # Add padding around the face to fit into 224x224 size
    padding = 30
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    
    # Ensure the bounding box does not go beyond the image boundaries
    x = max(x, 0)
    y = max(y, 0)
    
    # Crop the face region
    face_img = img[y:y+h, x:x+w]

    # Resize the face image to match the input size of your model
    target_size = (224, 224)
    face_img = cv2.resize(face_img, target_size)

    # Convert the face image to array and preprocess it for prediction
    face_array = image.img_to_array(face_img)
    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
    face_array /= 255.  # Normalize pixel values

    # Perform prediction
    prediction = model.predict(face_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction[0])

    # Map class index to class name
    class_names = train_generator.class_indices  # Assuming train_generator is defined
    class_name = [k for k, v in class_names.items() if v == predicted_class_index][0]

    print('Predicted class:', class_name)
else:
    print('No faces detected in the image.')