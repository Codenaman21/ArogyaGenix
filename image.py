from PIL import Image, ImageDraw, ImageFont

# Create a blank image with white background
width, height = 800, 600
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

# Load a font (you can use a system font or provide a .ttf file)
try:
    font = ImageFont.truetype("arial.ttf", 20)  # Use Arial font
except IOError:
    font = ImageFont.load_default()  # Fallback to default font

# Define the urine report data
data = {
    "Age": "32",
    "Gender": "M",
    "pH": "5.0",
    "Glucose": "+2",
    "Protein": "+3",
    "Ketones": "+3",
    "Bilirubin": "Trace",
    "Urobilinogen": "+3",
    "Nitrites": "Positive",
    "Specific Gravity": "1.006",
    "Diagnosis": "Kidney Disease"
}

# Define the position and spacing for the text
x, y = 50, 50
line_spacing = 30

# Draw the urine report data on the image
for key, value in data.items():
    text = f"{key}: {value}"
    draw.text((x, y), text, fill="black", font=font)
    y += line_spacing

# Save the image
image.save("urine_report.png")
print("Urine report image saved as 'urine_report.png'")