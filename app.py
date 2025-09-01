from flask import Flask, render_template, request
import os
import pytesseract
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html", title="Home - Visual Search OCR")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", title="Upload OCR", error="No file selected.")
        
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # OCR Extraction
        extracted_text = pytesseract.image_to_string(Image.open(filepath))
        
        return render_template("result.html", title="Result - Visual Search OCR", text=extracted_text)

    return render_template("upload.html", title="Upload OCR")

@app.route("/about")
def about():
    return render_template("base.html", title="About")

if __name__ == "__main__":
    app.run(debug=True)
