import torch
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
import cv2

class ImageClassifier:
    def __init__(self, model_path, class_names_path):
        self.model = torch.load(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.class_names = self.load_class_names(class_names_path)

    def load_class_names(self, file_path):
        with open(file_path, "r") as file:
            return file.read().split(",")

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def predict_image(self, image_path):
        data = self.preprocess_image(image_path)
        with torch.no_grad():
            prediction = self.model(data)
        index = torch.argmax(prediction).item()
        class_name = self.class_names[index]
        confidence_score = torch.softmax(prediction, dim=1)[0][index].item()
        return class_name, confidence_score

class GUI(tk.Tk):
    def __init__(self, classifier):
        super().__init__()

        self.classifier = classifier

        self.title("Image Classifier")
        self.geometry("400x400")

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        self.predict_button = tk.Button(self, text="Predict", command=self.predict_image)
        self.predict_button.pack(pady=10)

        self.quit_button = tk.Button(self, text="Quit", command=self.destroy)
        self.quit_button.pack(pady=10)

    def predict_image(self):
        img_path = tk.filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if img_path:
            class_name, confidence_score = self.classifier.predict_image(img_path)
            self.display_image(img_path)
            self.show_prediction(class_name, confidence_score)

    def display_image(self, img_path):
        image = Image.open(img_path)
        image = ImageTk.PhotoImage(image)
        self.image_label.config(image=image)
        self.image_label.image = image

    def show_prediction(self, class_name, confidence_score):
        result_text = f"Class: {class_name}\nConfidence: {confidence_score:.2%}"
        tk.messagebox.showinfo("Prediction", result_text)

def main():
    model_path = 'your_model.pth'  # Change to the actual path of your PyTorch model file
    class_names_path = 'classnames.txt'

    classifier = ImageClassifier(model_path, class_names_path)

    gui = GUI(classifier)
    gui.mainloop()

if __name__ == "__main__":
    main()
