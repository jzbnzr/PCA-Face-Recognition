import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.linalg import svd
import os
from sklearn.decomposition import PCA
import time 
import pickle
from PIL import Image, ImageTk

PROCESSED_DIR = "processed_faces"


if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

class FacePCAProcessor:
    def __init__(self, n_components=150):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
        
    def fit(self, database_images):
      
        # Convert torch tensor to numpy for PCA
        if isinstance(database_images, torch.Tensor):
            database_images = database_images.numpy()
            
        self.pca.fit(database_images)
        self.is_fitted = True
        
    def transform(self, images):
    
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform")
            
        # Convert to numpy if tensor
        if isinstance(images, torch.Tensor):
            images = images.numpy()
            
        # Transform the images
        transformed = self.pca.transform(images)
        
        # Convert back to torch tensor
        return torch.from_numpy(transformed).float()
    
    def inverse_transform(self, transformed_images):
       
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before inverse_transform")
            
        if isinstance(transformed_images, torch.Tensor):
            transformed_images = transformed_images.numpy()
            
        reconstructed = self.pca.inverse_transform(transformed_images)
        return torch.from_numpy(reconstructed).float()
    
    def save(self, filepath):
        
        with open(filepath, 'wb') as f:
            pickle.dump({'pca': self.pca, 'is_fitted': self.is_fitted}, f)
    
    @classmethod
    def load(cls, filepath):
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(n_components=data['pca'].n_components_)
        instance.pca = data['pca']
        instance.is_fitted = data['is_fitted']
        return instance



class StatsPanel:
    def __init__(self, master):
        self.frame = tk.Frame(master, relief=tk.RAISED, borderwidth=1)
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.total_images = tk.StringVar(value="Total Images: 0")
        self.total_celebrities = tk.StringVar(value="Total Celebrities: 0")
        
        tk.Label(self.frame, textvariable=self.total_images).pack(side=tk.LEFT, padx=5)
        tk.Label(self.frame, textvariable=self.total_celebrities).pack(side=tk.LEFT, padx=5)
        
    def update_stats(self):
        total_images = 0
        total_celebrities = 0
        
        for celebrity in os.listdir(PROCESSED_DIR):
            celebrity_path = os.path.join(PROCESSED_DIR, celebrity)
            if os.path.isdir(celebrity_path):
                total_celebrities += 1
                total_images += len([f for f in os.listdir(celebrity_path) if f.endswith('.jpg')])
        
        self.total_images.set(f"Total Images: {total_images}")
        self.total_celebrities.set(f"Total Celebrities: {total_celebrities}")

class ImageViewer:
    def __init__(self):
        self.window = None
        
    def show_images(self, celebrity):
        if self.window:
            self.window.destroy()
            
        self.window = tk.Toplevel()
        self.window.title(f"Images of {celebrity}")
        
        image_path = os.path.join(PROCESSED_DIR, celebrity)
        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"No images found for {celebrity}")
            return
            
        images = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
        rows = (len(images) + 4) // 5  # 5 images per row
        
        for i, img_file in enumerate(images):
            img = cv2.imread(os.path.join(image_path, img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            
            label = tk.Label(self.window, image=img)
            label.image = img  # Keep a reference
            label.grid(row=i//5, column=i%5, padx=5, pady=5)

def preprocess_images(images, dtype=torch.float32):
    images_array = np.array(images, dtype=np.float32)
    return torch.tensor(images_array, dtype=dtype)

global_pca_processor = None
global_transformed_database = None
global_labels = None


def precompute_pca_database(database_dir, save_path, n_components=100):
    """
    Precompute PCA components for the face database
    
    Args:
        database_dir: Directory containing processed face images
        save_path: Path to save the PCA model
        n_components: Number of PCA components to use
    Returns:
        Transformed database images and PCA processor
    """
    # Load all faces and labels
    database_images, labels = load_images_and_labels()
    
    if len(database_images) == 0:
        raise ValueError("No faces found in database")
    
    # Initialize and fit PCA
    pca_processor = FacePCAProcessor(n_components=n_components)
    pca_processor.fit(database_images)
    
    # Transform database images
    transformed_database = pca_processor.transform(database_images)
    
    # Save the PCA model
    pca_processor.save(save_path)
    
    return transformed_database, labels, pca_processor




def load_images_and_labels():
    images, labels = [], []
    for label in os.listdir(PROCESSED_DIR):
        label_path = os.path.join(PROCESSED_DIR, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.endswith(".jpg"):
                    image_path = os.path.join(label_path, file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        image = cv2.resize(image, (100, 100)).flatten()
                        images.append(image)
                        labels.append(label)
    return preprocess_images(images), labels



def recognize_face(face_image):
    """Modified recognize_face function that uses global PCA data"""
    global global_pca_processor, global_transformed_database, global_labels
    
    if global_pca_processor is None or global_transformed_database is None:
        return "No faces in database"
    
    try:
        # Resize and flatten the face image
        face_resized = cv2.resize(face_image, (100, 100)).flatten()
        
        # Preprocess and transform the input face
        face_tensor = preprocess_images([face_resized])
        transformed_face = global_pca_processor.transform(face_tensor)
        
        # Normalize the vectors
        transformed_face = F.normalize(transformed_face, p=2, dim=1)
        transformed_database = F.normalize(global_transformed_database, p=2, dim=1)
        
        # Calculate similarities
        similarities = F.cosine_similarity(transformed_face, transformed_database)
        
        if len(similarities) == 0:
            return "Error in recognition"
        
        # Get the most similar face
        max_similarity_idx = similarities.argmax().item()
        
        if max_similarity_idx >= len(global_labels):
            return "Recognition error"
        
        similarity_score = similarities[max_similarity_idx].item()
        
        if similarity_score < 0.6:  # Threshold for recognition
            return "Unknown Person"
        
        return f"{global_labels[max_similarity_idx]} ({similarity_score:.2%} match)"
        
    except Exception as e:
        print(f"Recognition error: {str(e)}")
        return "Recognition error"


def start_webcam_recognition():
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            break
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            face, face_rect = detect_and_recognize_face(frame)
            
            if face is not None and face_rect is not None:
                x, y, w, h = face_rect
                
                try:
                    result = recognize_face(face)
                except Exception as e:
                    result = "Recognition error"
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Put recognition result
                cv2.putText(frame, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow('Face Recognition', frame)
            
        except Exception as e:
            print(f"Error in recognition loop: {str(e)}")
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def load_images_and_labels():
    images, labels = [], []
    try:
        for label in os.listdir(PROCESSED_DIR):
            label_path = os.path.join(PROCESSED_DIR, label)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith(".jpg"):
                        image_path = os.path.join(label_path, file)
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            image = cv2.resize(image, (100, 100)).flatten()
                            images.append(image)
                            labels.append(label)
    except Exception as e:
        print(f"Error loading images: {str(e)}")
        return torch.tensor([]), []
        
    if not images:
        return torch.tensor([]), []
        
    return preprocess_images(images), labels

class LoadingDialog:
    def __init__(self, parent, title="Computing PCA"):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.transient(parent)
        self.top.grab_set()
        
        # Center the dialog
        window_width = 300
        window_height = 100
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.top.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Progress bar
        self.label = tk.Label(self.top, text="Initializing PCA...", pady=10)
        self.label.pack()
        
        self.progress = ttk.Progressbar(
            self.top, 
            orient="horizontal",
            length=200, 
            mode="determinate"
        )
        self.progress.pack(pady=10)
        
    def update_progress(self, value, text=None):
        self.progress["value"] = value
        if text:
            self.label.config(text=text)
        self.top.update()
        
    def destroy(self):
        self.top.destroy()

def initialize_pca_with_progress():
    """Initialize PCA with progress bar"""
    global global_pca_processor, global_transformed_database, global_labels
    
    loading_dialog = LoadingDialog(root)
    try:
        # Load images with progress
        loading_dialog.update_progress(0, "Loading images...")
        images, labels = [], []
        
        # Get total number of images first
        total_images = 0
        for label in os.listdir(PROCESSED_DIR):
            label_path = os.path.join(PROCESSED_DIR, label)
            if os.path.isdir(label_path):
                total_images += len([f for f in os.listdir(label_path) if f.endswith('.jpg')])
        
        if total_images == 0:
            loading_dialog.destroy()
            return
            
        current_images = 0
        for label in os.listdir(PROCESSED_DIR):
            label_path = os.path.join(PROCESSED_DIR, label)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith(".jpg"):
                        image_path = os.path.join(label_path, file)
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            image = cv2.resize(image, (100, 100)).flatten()
                            images.append(image)
                            labels.append(label)
                            current_images += 1
                            progress = (current_images / total_images) * 40
                            loading_dialog.update_progress(
                                progress, 
                                f"Loading images... ({current_images}/{total_images})"
                            )
        
        if not images:
            loading_dialog.destroy()
            return
            
        # Convert to tensor
        loading_dialog.update_progress(45, "Converting to tensor...")
        database_images = preprocess_images(images)
        
        # Initialize PCA
        loading_dialog.update_progress(50, "Initializing PCA...")
        pca_processor = FacePCAProcessor(n_components=150)
        
        # Fit PCA
        loading_dialog.update_progress(60, "Fitting PCA...")
        pca_processor.fit(database_images)
        
        # Transform database
        loading_dialog.update_progress(80, "Transforming database...")
        transformed_database = pca_processor.transform(database_images)
        
        # Save PCA model
        loading_dialog.update_progress(90, "Saving PCA model...")
        pca_processor.save(os.path.join(PROCESSED_DIR, "pca_model.pkl"))
        
        # Update global variables
        global_pca_processor = pca_processor
        global_transformed_database = transformed_database
        global_labels = labels
        
        loading_dialog.update_progress(100, "PCA computation complete!")
        root.after(1000, loading_dialog.destroy)  # Close after 1 second
        
    except Exception as e:
        messagebox.showerror("Error", f"Error initializing PCA: {str(e)}")
        loading_dialog.destroy()


def detect_and_recognize_face(frame):
    cascade_paths = [
        "/home/jahanzaib/anaconda3/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    ]
    
    face_cascade = None
    for path in cascade_paths:
        if os.path.exists(path):
            try:
                face_cascade = cv2.CascadeClassifier(path)
                if not face_cascade.empty():
                    break
            except:
                continue
    
    if face_cascade is None or face_cascade.empty():
        return None, None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))
        return face_resized, (x, y, w, h)
    return None, None

def start_webcam_recognition():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        face, face_rect = detect_and_recognize_face(frame)
        
        if face is not None:
            x, y, w, h = face_rect
            result = recognize_face(face)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Put recognition result
            cv2.putText(frame, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def upload_and_recognize():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
    )
    if not file_path:
        return
        
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Could not load image")
        return
        
    face, face_rect = detect_and_recognize_face(image)
    
    if face is not None:
        x, y, w, h = face_rect
        result = recognize_face(face)
        
        # Draw rectangle and result on image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Recognition Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        messagebox.showerror("Error", "No face detected in the image")



def capture_faces(label):
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            break
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera. Please check your camera connection.")
        return
        
    captured_faces = []

    def detect_and_resize_face(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cascade_paths = [
            "/home/jahanzaib/anaconda3/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
        ]
        
        face_cascade = None
        for path in cascade_paths:
            if os.path.exists(path):
                try:
                    face_cascade = cv2.CascadeClassifier(path)
                    if not face_cascade.empty():
                        break
                except:
                    continue
                
        if face_cascade is None or face_cascade.empty():
            messagebox.showerror("Error", "Could not find or load face cascade file. Please check the file path.")
            return None

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))
            return face_resized
        return None

    count = 0
    try:
        while count < 10:
            ret, frame = cap.read()
            if not ret:
                continue

            if frame is None:
                messagebox.showerror("Error", "Cannot read from camera. Please check your camera connection.")
                break

            face = detect_and_resize_face(frame)
            if face is not None:
                captured_faces.append(face)
                count += 1
                cv2.imshow("Captured Face", face)
                cv2.waitKey(1000)

            progress_text = f"Captured {count}/10 Faces"
            instruction_text = "Press SPACE to capture (Press Q to quit)"
            #cv2.putText(frame, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, progress_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           
           
        
     
            
            cv2.imshow("Webcam", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                captured_faces.append(face)
                count += 1
                cv2.imshow("Captured Face", face)
                cv2.waitKey(1000)
                time.sleep(2)  # Brief pause to show the captured face
            
            elif key == ord("q"):
                break


            """time.sleep(1)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break"""
            


    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(captured_faces) == 10:
        save_faces(label, captured_faces)
        messagebox.showinfo("Success", f"Captured 10 faces for {label}.")
        stats_panel.update_stats()
    else:
        messagebox.showerror("Error", "Failed to capture 10 faces.")

def save_faces(label, faces):
    label_dir = os.path.join(PROCESSED_DIR, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    for i, face in enumerate(faces):
        file_path = os.path.join(label_dir, f"{label}_{i}.jpg")
        cv2.imwrite(file_path, face)

    update_records()

def delete_face(label):
    label_dir = os.path.join(PROCESSED_DIR, label)
    if os.path.exists(label_dir):
        for file in os.listdir(label_dir):
            os.remove(os.path.join(label_dir, file))
        os.rmdir(label_dir)
        messagebox.showinfo("Success", f"Deleted all faces for {label}")
        stats_panel.update_stats()
    else:
        messagebox.showerror("Error", f"No record found for {label}")
    
    update_records()

def add_face_gui():
    label = simple_input_dialog("Add Face", "Enter the celebrity name:")
    if label:
        capture_faces(label)

def delete_face_gui():
    try:
        selected = records_list.get(records_list.curselection())
        if selected:
            confirm = messagebox.askyesno("Confirm", f"Are you sure you want to delete {selected}?")
            if confirm:
                delete_face(selected)
    except:
        messagebox.showerror("Error", "Please select a celebrity to delete")

def view_mean_face():
    images, _ = load_images_and_labels()
    if len(images) > 0:
        mean_face = compute_mean_face(images)
        if mean_face is not None:
            mean_face_image = mean_face.numpy().reshape(100, 100).astype(np.uint8)
            cv2.imshow("Mean Face", mean_face_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        messagebox.showinfo("Info", "No faces in database to compute mean face")

def compute_mean_face(data):
    if len(data) == 0:
        return None
    return data.mean(dim=0)

def update_pca_database():
    """Update the PCA database and store results in global variables"""
    global global_pca_processor, global_transformed_database, global_labels
    
    try:
        transformed_db, labels, pca_processor = precompute_pca_database(
            database_dir=PROCESSED_DIR,
            save_path=os.path.join(PROCESSED_DIR, "pca_model.pkl"),
            n_components=150
        )
        
        global_pca_processor = pca_processor
        global_transformed_database = transformed_db
        global_labels = labels
        
    except Exception as e:
        print(f"Error computing PCA: {str(e)}")
        global_pca_processor = None
        global_transformed_database = None
        global_labels = None

def update_records():
    records_list.delete(0, tk.END)
    for label in os.listdir(PROCESSED_DIR):
        if os.path.isdir(os.path.join(PROCESSED_DIR, label)):
            records_list.insert(tk.END, label)
    stats_panel.update_stats()
    
    # Update PCA database whenever records are updated
    update_pca_database()

def simple_input_dialog(title, prompt):
    def on_submit():
        nonlocal result
        result = entry.get()
        dialog.destroy()

    dialog = tk.Toplevel(root)
    dialog.title(title)
    tk.Label(dialog, text=prompt).pack(padx=10, pady=10)
    entry = tk.Entry(dialog)
    entry.pack(padx=10, pady=10)
    tk.Button(dialog, text="Submit", command=on_submit).pack(pady=10)

    result = None
    dialog.transient(root)
    dialog.grab_set()
    root.wait_window(dialog)
    return result


    """Update the PCA database and store results in global variables"""
    global global_pca_processor, global_transformed_database, global_labels
    
    try:
        transformed_db, labels, pca_processor = precompute_pca_database(
            database_dir=PROCESSED_DIR,
            save_path=os.path.join(PROCESSED_DIR, "pca_model.pkl"),
            n_components=150
        )
        
        global_pca_processor = pca_processor
        global_transformed_database = transformed_db
        global_labels = labels
        
    except Exception as e:
        print(f"Error computing PCA: {str(e)}")
        global_pca_processor = None
        global_transformed_database = None
        global_labels = None

def on_celebrity_double_click(event):
    selection = records_list.curselection()
    if selection:
        celebrity = records_list.get(selection[0])
        image_viewer.show_images(celebrity)


def initialize_gui():
    global root, stats_panel, image_viewer, records_list
    
    root = tk.Tk()
    root.title("Face Recognition Using PCA")
    
    # Create stats panel
    stats_panel = StatsPanel(root)
    image_viewer = ImageViewer()
    
    # Buttons frame
    frame = tk.Frame(root)
    frame.pack(pady=20)
    
    tk.Button(frame, text="Add Face", command=add_face_gui).grid(row=0, column=0, padx=5)
    tk.Button(frame, text="Delete Face", command=delete_face_gui).grid(row=0, column=1, padx=5)
    tk.Button(frame, text="View Mean Face", command=view_mean_face).grid(row=0, column=2, padx=5)
    tk.Button(frame, text="Upload Image", command=upload_and_recognize).grid(row=0, column=3, padx=5)
    tk.Button(frame, text="Start Webcam", command=start_webcam_recognition).grid(row=0, column=4, padx=5)
    
    tk.Label(root, text="Celebrities in Database\n(Double-click to view images)").pack(pady=10)
    records_list = tk.Listbox(root, width=50)
    records_list.pack()
    
    # Bind double-click event
    records_list.bind('<Double-Button-1>', on_celebrity_double_click)
    
    update_records()
    
    # Initialize PCA after GUI is created
    root.after(100, initialize_pca_with_progress)
    
    return root


if __name__ == "__main__":
    root = initialize_gui()
    root.mainloop()