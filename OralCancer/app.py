"""
Flask web app for OSCC-CNN oral cancer detection.

Key features and security hardening:
  1. /submit only accepts POST requests and requires a valid file upload.
  2. Enforces a 16 MB maximum upload limit (MAX_CONTENT_LENGTH) to prevent DoS.
  3. Validates file extension AND verifies image data integrity using Pillow.
  4. Uses werkzeug secure_filename() + UUID prefix to prevent directory traversal.
  5. Automatically cleans up old temporary uploads to prevent disk exhaustion.
  6. Immediately deletes corrupt/failed uploads in exception handlers.
  7. Secret key and debug mode configurable via environment variables.
"""

import json
import os
import time
import uuid
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image, ImageFile
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

ImageFile.LOAD_TRUNCATED_IMAGES = True

from main import getPrediction

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "images"
SAMPLES_DIR = BASE_DIR / "static" / "samples"
SAMPLES_MANIFEST = SAMPLES_DIR / "manifest.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
MAX_FILE_AGE_SECONDS = 3600  # Prune uploads older than 1 hour

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "oscc-cnn-dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_image(file_path: Path) -> bool:
    """Verifies that the file is a genuine, uncorrupted image using Pillow."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def prune_old_uploads():
    """Removes uploaded images older than MAX_FILE_AGE_SECONDS to prevent storage leaks."""
    if not UPLOAD_FOLDER.exists():
        return
    now = time.time()
    for item in UPLOAD_FOLDER.iterdir():
        if item.is_file() and not item.name.startswith("."):
            try:
                if now - item.stat().st_mtime > MAX_FILE_AGE_SECONDS:
                    item.unlink(missing_ok=True)
            except OSError:
                pass


@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    flash("The uploaded image is too large. Please select an image under 16 MB.")
    return redirect(url_for("home"))


@app.route("/home")
@app.route("/", methods=["GET"])
def home():
    sample_images = []
    if SAMPLES_MANIFEST.exists():
        try:
            with open(SAMPLES_MANIFEST, encoding="utf-8") as f:
                manifest = json.load(f)
            for entry in manifest.get("images", []):
                rel_file = entry.get("file", "").replace("\\", "/")
                file_path = SAMPLES_DIR / rel_file
                if file_path.exists():
                    sample_images.append({
                        "url": url_for("static", filename=f"samples/{rel_file}"),
                        "label": entry["label"],
                        "filename": Path(rel_file).name,
                        "rel_path": rel_file,
                    })
        except Exception:
            pass
    return render_template("index.html", sample_images=sample_images)


@app.route("/samples")
def samples():
    dataset_info = {}
    normal_images = []
    oscc_images = []

    if SAMPLES_MANIFEST.exists():
        with open(SAMPLES_MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f)

        dataset_info = manifest.get("dataset", {})
        for entry in manifest.get("images", []):
            rel_file = entry.get("file", "").replace("\\", "/")
            file_path = SAMPLES_DIR / rel_file
            if not file_path.exists():
                continue
            item = {
                "url": url_for("static", filename=f"samples/{rel_file}"),
                "label": entry["label"],
                "filename": Path(rel_file).name,
                "rel_path": rel_file,
                "magnification": "100×" if "100x" in rel_file.lower() else ("400×" if "400x" in rel_file.lower() else "H&E Biopsy"),
            }
            if entry["label"].lower() == "normal":
                normal_images.append(item)
            else:
                oscc_images.append(item)

    return render_template(
        "samples.html",
        dataset_info=dataset_info,
        normal_images=normal_images,
        oscc_images=oscc_images,
        has_images=bool(normal_images or oscc_images),
    )


@app.route("/submit", methods=["POST"])
def get_output():
    sample_choice = request.form.get("sample_choice")
    image_file = request.files.get("my_image")

    # Path 1: User selected a preloaded reference sample
    if sample_choice:
        rel_parts = Path(sample_choice).parts
        safe_rel = Path(*[secure_filename(p) for p in rel_parts])
        sample_path = SAMPLES_DIR / safe_rel

        if sample_path.exists() and is_valid_image(sample_path):
            try:
                result = getPrediction(str(sample_path))
                return render_template(
                    "index_report.html",
                    prediction=result["label"],
                    confidence=result["confidence"],
                    img_url=url_for("static", filename=f"samples/{safe_rel}"),
                )
            except Exception as exc:
                flash(f"Could not analyze reference sample: {exc}")
                return redirect(url_for("home"))
        else:
            flash("The selected sample image was not found or is unreadable.")
            return redirect(url_for("home"))

    # Path 2: User uploaded their own image
    if image_file is None or image_file.filename == "":
        flash("Please upload a histopathology image or select a sample before analyzing.")
        return redirect(url_for("home"))

    if not allowed_file(image_file.filename):
        flash("Unsupported file format! Please upload a valid image (.png, .jpg, .jpeg, .bmp, .tif).")
        return redirect(url_for("home"))

    # Housekeeping: prune older uploads to keep disk footprint minimal
    prune_old_uploads()

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{secure_filename(image_file.filename)}"
    img_path = UPLOAD_FOLDER / safe_name
    image_file.save(img_path)

    # Strictly validate image data integrity using Pillow
    if not is_valid_image(img_path):
        img_path.unlink(missing_ok=True)
        flash("Invalid file content! The file you uploaded is not a readable or valid image.")
        return redirect(url_for("home"))

    try:
        result = getPrediction(str(img_path))
    except Exception as exc:  # noqa: BLE001
        img_path.unlink(missing_ok=True)
        flash(f"Could not analyze that image: {exc}")
        return redirect(url_for("home"))

    return render_template(
        "index_report.html",
        prediction=result["label"],
        confidence=result["confidence"],
        img_url=url_for("static", filename=f"images/{safe_name}"),
    )


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("true", "1")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5001"))
    app.run(host=host, port=port, debug=debug_mode)
