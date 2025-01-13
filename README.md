# Face Recognition Using PCA

A real-time face recognition system using Principal Component Analysis (PCA) with a graphical user interface built in Python. The system allows users to manage a face database, capture faces through webcam, and perform real-time face recognition.

## Features

- Real-time face detection and recognition using webcam
- Face recognition from uploaded images
- Database management for face data
- Interactive GUI for all operations
- PCA-based face recognition algorithm
- Progress tracking for intensive operations
- Face database statistics
- Mean face visualization

## Requirements

```
Python 3.7+
OpenCV (cv2)
NumPy
PyTorch
scikit-learn
Pillow
tkinter (usually comes with Python)
```

## Installation

1. Clone the repository:
```bash
git clone PCA-Face-Recognition
cd PCA-Face-Recognition
```

2. Install required packages:
```bash
pip install opencv-python numpy torch scikit-learn pillow
```

3. Ensure you have the Haar Cascade file. The system looks for it in the following locations:
- `/home/[user]/anaconda3/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`
- `/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`
- `/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`
- In the same directory as the script

## Dataset 

The dataset has been obtained from Kaggle: 
https://www.kaggle.com/api/v1/datasets/download/vasukipatel/face-recognition-dataset

The code to download and process the dataset can be found in download_dataset.ipynb



## Usage

1. Run the main script:
```bash
python main.py
```

2. The GUI provides the following options:

- **Add Face**: Capture faces through webcam for a new person
- **Delete Face**: Remove a person from the database
- **View Mean Face**: Display the average face from the database
- **Upload Image**: Recognize faces in an uploaded image
- **Start Webcam**: Begin real-time face recognition

3. Adding a new face:
   - Click "Add Face"
   - Enter the person's name
   - The system will capture 10 faces automatically
   - Wait for the capture process to complete

4. Real-time recognition:
   - Click "Start Webcam"
   - Position face in front of camera
   - Press 'q' to quit

## Project Structure

```
face-recognition-pca/
│
├── face_recognition.py      # Main application file
├── processed_faces/         # Directory for stored face data
│   ├── [person_name]/      # Individual directories for each person
│   └── pca_model.pkl       # Saved PCA model
│
└── README.md               # This file
```

## Implementation Details

### Face Detection
- Uses OpenCV's Haar Cascade Classifier
- Processes frames in real-time
- Includes face detection parameter tuning

### PCA Implementation
- Reduces dimensionality to 150 components
- Implements batch processing for large datasets
- Includes progress tracking during computation
- Caches transformed database

### GUI Features
- Interactive list of people in database
- Double-click to view captured faces
- Progress indicators for long operations
- Real-time statistics display

## Troubleshooting

1. **Camera not detected**
   - The system tries camera indices 0, 1, and 2
   - Ensure your camera is properly connected
   - Check camera permissions

2. **Face detection issues**
   - Ensure good lighting conditions
   - Face should be clearly visible and front-facing
   - Adjust distance from camera

3. **PCA computation errors**
   - Ensure sufficient memory is available
   - Check if database directory has proper permissions
   - Verify face images are properly formatted

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV team for the Haar Cascade implementation
- PyTorch team for the tensor operations framework
- scikit-learn team for PCA implementation

## Contact

For any queries or issues, please open an issue in the repository.