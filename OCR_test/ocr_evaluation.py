# Import necessary libraries
from PIL import Image, ImageEnhance
import pytesseract
import easyocr
from paddleocr import PaddleOCR
import time

# Path to the original image and where to save the enhanced image
original_image_path = r'D:\Python test\OCR_test\OCR_test_image.jpg'
enhanced_image_path = r'D:\Python test\OCR_test\enhanced_image.jpg'

# Function to enhance the image (contrast and grayscale)
def enhance_image(image_path, output_path):
    # Open the image
    img = Image.open(image_path)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # Adjust contrast level as needed
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Save the enhanced image
    img.save(output_path)
    print(f"Enhanced image saved at {output_path}")

# Tesseract OCR function
def tesseract_ocr(image_path):
    image = Image.open(image_path)
    start_time = time.time()
    text = pytesseract.image_to_string(image)
    elapsed_time = time.time() - start_time
    return text, elapsed_time

# EasyOCR function
def easyocr_ocr(image_path):
    reader = easyocr.Reader(['en'])  # Add other languages if needed
    start_time = time.time()
    result = reader.readtext(image_path)
    elapsed_time = time.time() - start_time
    detected_text = " ".join([text for _, text, _ in result])
    return detected_text, elapsed_time

# PaddleOCR function
def paddleocr_ocr(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize PaddleOCR
    start_time = time.time()
    result = ocr.ocr(image_path, det=True, rec=True)  # Ensure detection and recognition are both enabled
    elapsed_time = time.time() - start_time
    # Flatten the result and get the recognized text
    detected_text = " ".join([res[1][0] for line in result for res in line])  # Correct way to access text
    return detected_text, elapsed_time

# Function to print the OCR results
def print_results():
    # Enhance the original image before OCR
    enhance_image(original_image_path, enhanced_image_path)

    print("\nEvaluating Tesseract OCR:")
    try:
        tesseract_result, tesseract_time = tesseract_ocr(enhanced_image_path)
        print(f"Tesseract OCR Result: {tesseract_result}\nTime taken: {tesseract_time:.2f} seconds")
    except Exception as e:
        print(f"Tesseract OCR failed: {e}")

    print("\nEvaluating EasyOCR:")
    try:
        easyocr_result, easyocr_time = easyocr_ocr(enhanced_image_path)
        print(f"EasyOCR Result: {easyocr_result}\nTime taken: {easyocr_time:.2f} seconds")
    except Exception as e:
        print(f"EasyOCR failed: {e}")

    print("\nEvaluating PaddleOCR:")
    try:
        paddleocr_result, paddleocr_time = paddleocr_ocr(enhanced_image_path)
        print(f"PaddleOCR Result: {paddleocr_result}\nTime taken: {paddleocr_time:.2f} seconds")
    except Exception as e:
        print(f"PaddleOCR failed: {e}")

# Run the evaluation
if __name__ == "__main__":
    print_results()
