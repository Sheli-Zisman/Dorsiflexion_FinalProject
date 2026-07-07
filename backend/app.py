from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import os
import glob
import uuid
import shutil
import yaml

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import deeplabcut


# ============================================================
# Paths
# ============================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

DLC_PROJECT_PATH = os.path.join(PROJECT_DIR, "training_cropped_50")
CONFIG_PATH = os.path.join(DLC_PROJECT_PATH, "config.yaml")

UPLOAD_FOLDER = os.path.join(BACKEND_DIR, "uploads")
ANNOTATED_FOLDER = os.path.join(BACKEND_DIR, "annotated")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# ============================================================
# Flask setup
# ============================================================

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ============================================================
# Confidence thresholds
# ============================================================

MIN_SHIN_CONF = 0.2
MIN_ANKLE_CONF = 0.4
MIN_FIFTH_CONF = 0.4


# ============================================================
# Helper functions
# ============================================================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def update_config_project_path():
    """
    Make sure config.yaml points to the local DLC project path.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["project_path"] = DLC_PROJECT_PATH

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def convert_upload_to_png(file_storage, output_path, target_size=1024):
    """
    Saves uploaded PNG/JPG/JPEG as corrected 1024x1024 PNG.
    Matches the preprocessing used for training images:
    fixes phone EXIF orientation, converts to RGB, resizes without cutting,
    and centers the image on a white 1024x1024 canvas.
    """
    image = Image.open(file_storage.stream)

    # Important for phone images, especially iPhone/Safari
    image = ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    w, h = image.size

    # Resize so the larger side becomes 1024
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Put on 1024x1024 white canvas without cutting
    canvas = Image.new("RGB", (target_size, target_size), "white")

    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2

    canvas.paste(image, (x, y))

    canvas.save(output_path, optimize=True)

def calculate_angle(shin, ankle, fifth):
    """
    Calculates the angle at ankle between:
    ankle -> shin
    ankle -> fifth
    """
    v1 = np.array([shin[0] - ankle[0], shin[1] - ankle[1]])
    v2 = np.array([fifth[0] - ankle[0], fifth[1] - ankle[1]])

    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm == 0:
        return np.nan

    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def save_annotated_image(
    img_path,
    shin,
    ankle,
    fifth,
    shin_conf,
    ankle_conf,
    fifth_conf,
    angle,
    valid,
    out_path
):
    img = Image.open(img_path)

    plt.figure(figsize=(7, 6))
    plt.imshow(img)

    # Points
    plt.scatter(shin[0], shin[1], s=80, label=f"shin {shin_conf:.2f}")
    plt.scatter(ankle[0], ankle[1], s=80, label=f"ankle {ankle_conf:.2f}")
    plt.scatter(fifth[0], fifth[1], s=80, label=f"fifth {fifth_conf:.2f}")

    # Labels
    plt.text(shin[0] + 5, shin[1] + 5, f"shin\n{shin_conf:.2f}", fontsize=9)
    plt.text(ankle[0] + 5, ankle[1] + 5, f"ankle\n{ankle_conf:.2f}", fontsize=9)
    plt.text(fifth[0] + 5, fifth[1] + 5, f"fifth\n{fifth_conf:.2f}", fontsize=9)

    # Angle lines
    plt.plot([ankle[0], shin[0]], [ankle[1], shin[1]], linewidth=2)
    plt.plot([ankle[0], fifth[0]], [ankle[1], fifth[1]], linewidth=2)

    filename = os.path.basename(img_path)

    if valid and not np.isnan(angle):
        plt.title(f"{filename}\nAngle = {angle:.2f}°")
    else:
        plt.title(f"{filename}\nLow confidence - angle not used")

    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_uploaded_pair(image1_path, image2_path, filename1, filename2):
    """
    Runs DeepLabCut on two uploaded images and returns angles + annotated image names.
    """

    update_config_project_path()

    request_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(UPLOAD_FOLDER, f"request_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)

    temp_img1 = os.path.join(temp_dir, filename1)
    temp_img2 = os.path.join(temp_dir, filename2)

    shutil.copy(image1_path, temp_img1)
    shutil.copy(image2_path, temp_img2)

    # Remove old prediction files in request folder
    for f in glob.glob(os.path.join(temp_dir, "*.h5")):
        os.remove(f)

    deeplabcut.analyze_time_lapse_frames(
        CONFIG_PATH,
        directory=temp_dir,
        frametype=".png",
        shuffle=4,
        trainingsetindex=0
    )

    pred_files = glob.glob(os.path.join(temp_dir, "*.h5"))

    if not pred_files:
        raise FileNotFoundError("DeepLabCut did not create a prediction .h5 file.")

    pred_file = pred_files[0]
    df = pd.read_hdf(pred_file)

    scorer = df.columns.get_level_values("scorer")[0]
    individual = df.columns.get_level_values("individuals")[0]

    results = {}

    for img_path, row in df.iterrows():
        base_name = os.path.basename(img_path)

        shin = (
            row[(scorer, individual, "shin", "x")],
            row[(scorer, individual, "shin", "y")]
        )

        ankle = (
            row[(scorer, individual, "ankle", "x")],
            row[(scorer, individual, "ankle", "y")]
        )

        fifth = (
            row[(scorer, individual, "fifth", "x")],
            row[(scorer, individual, "fifth", "y")]
        )

        shin_conf = float(row[(scorer, individual, "shin", "likelihood")])
        ankle_conf = float(row[(scorer, individual, "ankle", "likelihood")])
        fifth_conf = float(row[(scorer, individual, "fifth", "likelihood")])

        valid = (
            shin_conf >= MIN_SHIN_CONF and
            ankle_conf >= MIN_ANKLE_CONF and
            fifth_conf >= MIN_FIFTH_CONF
        )

        angle = calculate_angle(shin, ankle, fifth) if valid else np.nan

        annotated_name = os.path.splitext(base_name)[0] + "_annotated.png"
        annotated_path = os.path.join(ANNOTATED_FOLDER, annotated_name)

        save_annotated_image(
            img_path=img_path,
            shin=shin,
            ankle=ankle,
            fifth=fifth,
            shin_conf=shin_conf,
            ankle_conf=ankle_conf,
            fifth_conf=fifth_conf,
            angle=angle,
            valid=valid,
            out_path=annotated_path
        )

        results[base_name] = {
            "angle": float(angle) if valid else None,
            "valid": bool(valid),
            "shin_conf": shin_conf,
            "ankle_conf": ankle_conf,
            "fifth_conf": fifth_conf,
            "annotated_name": annotated_name
        }

    if filename1 not in results or filename2 not in results:
        raise RuntimeError("Could not match prediction results to uploaded image filenames.")

    return results[filename1], results[filename2]


# ============================================================
# API routes
# ============================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Server is running",
        "dlc_project_path": DLC_PROJECT_PATH,
        "config_exists": os.path.exists(CONFIG_PATH),
        "model_folder_exists": os.path.exists(os.path.join(DLC_PROJECT_PATH, "dlc-models-pytorch")),
        "frontend_exists": os.path.exists(os.path.join(FRONTEND_DIR, "index.html"))
    })


@app.route("/api/analyze-range", methods=["POST", "OPTIONS"])
def analyze_range_of_motion():
    try:
        if request.method == "OPTIONS":
            return ("", 204)

        if "image1" not in request.files or "image2" not in request.files:
            return jsonify({"error": "Two images are required"}), 400

        file1 = request.files["image1"]
        file2 = request.files["image2"]

        if file1.filename == "" or file2.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            return jsonify({
                "error": "Invalid file type. Only PNG, JPG, and JPEG are allowed"
            }), 400

        if not os.path.exists(CONFIG_PATH):
            return jsonify({"error": f"config.yaml not found: {CONFIG_PATH}"}), 500

        if not os.path.exists(os.path.join(DLC_PROJECT_PATH, "dlc-models-pytorch")):
            return jsonify({"error": "dlc-models-pytorch folder not found"}), 500

        unique_id = str(uuid.uuid4())[:8]

        filename1 = f"upload_{unique_id}_image1.png"
        filename2 = f"upload_{unique_id}_image2.png"

        filepath1 = os.path.join(UPLOAD_FOLDER, filename1)
        filepath2 = os.path.join(UPLOAD_FOLDER, filename2)

        convert_upload_to_png(file1, filepath1)
        convert_upload_to_png(file2, filepath2)

        result1, result2 = analyze_uploaded_pair(
            image1_path=filepath1,
            image2_path=filepath2,
            filename1=filename1,
            filename2=filename2
        )


        print("\n--- Confidence Debug ---")
        print("Image 1 valid:", result1["valid"])
        print("Image 1 angle:", result1["angle"])
        print(
            "Image 1 confidence:",
            "shin =", result1["shin_conf"],
            "ankle =", result1["ankle_conf"],
            "fifth =", result1["fifth_conf"]
        )

        print("Image 2 valid:", result2["valid"])
        print("Image 2 angle:", result2["angle"])
        print(
            "Image 2 confidence:",
            "shin =", result2["shin_conf"],
            "ankle =", result2["ankle_conf"],
            "fifth =", result2["fifth_conf"]
        )
        print("------------------------\n")

        if not result1["valid"] or not result2["valid"]:
            return jsonify({
                "error": "Low confidence detection. Please retake one or both images.",
                "image1_valid": result1["valid"],
                "image2_valid": result2["valid"],
                "image1_confidence": {
                    "shin": result1["shin_conf"],
                    "ankle": result1["ankle_conf"],
                    "fifth": result1["fifth_conf"],
                },
                "image2_confidence": {
                    "shin": result2["shin_conf"],
                    "ankle": result2["ankle_conf"],
                    "fifth": result2["fifth_conf"],
                }
            }), 422

        angle1 = result1["angle"]
        angle2 = result2["angle"]
        range_of_motion = abs(angle1 - angle2)

        response = {
            "angle1": float(angle1),
            "angle2": float(angle2),
            "range_of_motion": float(range_of_motion),

            "filename1": filename1,
            "filename2": filename2,

            "annotated1_exists": os.path.exists(
                os.path.join(ANNOTATED_FOLDER, result1["annotated_name"])
            ),
            "annotated2_exists": os.path.exists(
                os.path.join(ANNOTATED_FOLDER, result2["annotated_name"])
            ),

            "image1_confidence": {
                "shin": result1["shin_conf"],
                "ankle": result1["ankle_conf"],
                "fifth": result1["fifth_conf"],
            },
            "image2_confidence": {
                "shin": result2["shin_conf"],
                "ankle": result2["ankle_conf"],
                "fifth": result2["fifth_conf"],
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/annotated/<filename>")
def get_annotated_image(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)


# ============================================================
# Frontend routes
# ============================================================

@app.route("/")
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def serve_static_files(path):
    file_path = os.path.join(FRONTEND_DIR, path)

    if os.path.exists(file_path):
        return send_from_directory(FRONTEND_DIR, path)

    return send_from_directory(FRONTEND_DIR, "index.html")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Starting server with DeepLabCut 3 / PyTorch model...")
    print("Project directory:", PROJECT_DIR)
    print("Frontend directory:", FRONTEND_DIR)
    print("DLC project path:", DLC_PROJECT_PATH)
    print("Config path:", CONFIG_PATH)
    print("Upload folder:", UPLOAD_FOLDER)
    print("Annotated folder:", ANNOTATED_FOLDER)

    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)