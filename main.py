import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from fpdf import FPDF
from PIL import Image as PILImage
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
# Load the trained model
model = load_model('brain_tumor_detection.h5')

# Define the classes
classes = ['no', 'yes']

def generate_medical_report(class_prediction, confidence):
    report = f"Medical Report\n\n"
    report += f"Prediction: {class_prediction}\n"
    report += f"Confidence: {confidence * 100:.2f}%\n"

    if class_prediction == 'yes':
        report += "\nFindings: The model has detected a brain tumor. Further medical evaluation is recommended."
    else:
        report += "\nFindings: No brain tumor detected. Follow-up examinations may be scheduled for confirmation."

    # Add date and time to the report
    report += f"\nReport Generated Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return report
def preprocess_image(image):
    # Preprocess the input image
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize pixel values to be between 0 and 1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Duplicate the processed image to provide two inputs for the model
    processed_images = [processed_image, processed_image]

    # Make predictions using the loaded model
    predictions = model.predict(processed_images)
    print(predictions)
    # Get the class with the highest probability for each output
    predicted_class1 = np.argmax(predictions[0][0])  # Assuming binary classification for output 1
    predicted_class2 = np.argmax(predictions[1][0])  # Assuming binary classification for output 2

    # Assuming binary classification, you can choose the class with higher confidence
    if predictions[0][0][predicted_class1] > predictions[1][0][predicted_class2]:
        final_class_prediction = classes[predicted_class1]
        final_confidence = predictions[0][0][predicted_class1]
    else:
        final_class_prediction = classes[predicted_class2]
        final_confidence = predictions[1][0][predicted_class2]

    return final_class_prediction, final_confidence



def main():
    st.title("Brain Tumor Detection App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a brain MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Perform prediction
        class_prediction, confidence = predict(image)
        medical_report = generate_medical_report(class_prediction, confidence)
        st.markdown(medical_report)

        # Save the medical report as a PDF
        pdf = FPDF()
        pdf.add_page()

        # Convert the OpenCV image to PIL format
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image_path = "image.png"  # Save the PIL image to a temporary file
        pil_image.save(pil_image_path)

        # Add the image to the PDF
        pdf.image(pil_image_path, x=10, y=pdf.get_y(), w=0, h=100)
        pdf.ln(100)

        # Add the medical report to the PDF
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, medical_report)

        # Add "MSM Labs" watermark in cross form
        pdf.set_text_color(128, 128, 128)  # Gray color for watermark
        pdf.set_font("Arial", style='I', size=30)
        pdf.ln(10)  # Move down 10 units before adding watermark

        # Draw horizontal line
        pdf.cell(0, 5, "MSM Labs", ln=True, align='C')
        # Draw vertical line
        pdf.cell(10)
        pdf.cell(0, 5, "", ln=True)

        # Save the PDF
        pdf_output = "medical_report.pdf"
        pdf.output(pdf_output)

        st.success(f"Medical report with image and watermarks saved as {pdf_output}")


if __name__ == '__main__':
    main()



