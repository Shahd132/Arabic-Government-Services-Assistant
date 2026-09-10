import os
import sys
import uuid
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    send_from_directory
)

from flask_session import Session
from werkzeug.utils import secure_filename


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT_DIR / "05_pipeline_eval_person5" / "src")
)

sys.path.insert(
    0,
    str(ROOT_DIR / "shared")
)


# ============================================================
# ENVIRONMENT
# ============================================================

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")


# ============================================================
# IMPORT YOUR GRAPH
# ============================================================

from graph import run, reset_conversation


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.urandom(24)

# Session configuration
app.config["SESSION_TYPE"] = "filesystem"

# Upload folder
app.config["UPLOAD_FOLDER"] = os.path.join(
    app.root_path,
    "uploads"
)

# Maximum upload size = 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# Initialize Flask-Session
Session(app)


# ============================================================
# CREATE UPLOAD FOLDER
# ============================================================

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)


# ============================================================
# ALLOWED FILE EXTENSIONS
# ============================================================

ALLOWED_EXTENSIONS = {
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "bmp",
    "tiff",
    "webp"
}


def allowed_file(filename):
    """
    Check whether the uploaded file has an allowed extension.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# SERVE UPLOADED FILES
# ============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# ASK / PROCESS QUESTION
# ============================================================

@app.route("/ask", methods=["POST"])
def ask():

    # --------------------------------------------------------
    # GET QUESTION
    # --------------------------------------------------------

    question = request.form.get("question")

    # If request is JSON
    if not question and request.is_json:

        data = request.get_json(silent=True) or {}

        question = data.get("question")


    # --------------------------------------------------------
    # CHECK QUESTION
    # --------------------------------------------------------

    if not question:

        return jsonify({
            "error": "Please enter a question."
        }), 400


    # --------------------------------------------------------
    # CHECK UPLOADED FILE
    # --------------------------------------------------------

    uploaded_file = request.files.get("document")

    document_path = None
    safe_filename = None
    image_url = None


    # --------------------------------------------------------
    # SAVE UPLOADED FILE
    # --------------------------------------------------------

    if uploaded_file and uploaded_file.filename:

        if not allowed_file(uploaded_file.filename):

            return jsonify({
                "error": "File type is not allowed."
            }), 400


        # Secure original filename
        filename = secure_filename(
            uploaded_file.filename
        )


        # Generate unique ID
        unique_id = str(uuid.uuid4())[:8]


        # Example:
        # abc12345_image.jpg

        safe_filename = f"{unique_id}_{filename}"


        # Full path on PC
        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            safe_filename
        )


        # Save file
        uploaded_file.save(filepath)


        # Path used by your pipeline
        document_path = filepath


        # URL that browser can access
        image_url = f"/uploads/{safe_filename}"


    # --------------------------------------------------------
    # CREATE / GET SESSION ID
    # --------------------------------------------------------

    session_id = session.get("session_id")


    if not session_id:

        session_id = str(uuid.uuid4())

        session["session_id"] = session_id


    # --------------------------------------------------------
    # RUN YOUR AI PIPELINE
    # --------------------------------------------------------

    try:

        result = run(
            question,
            document_path=document_path,
            session_id=session_id
        )


        # ----------------------------------------------------
        # GET RESULTS
        # ----------------------------------------------------

        answer = result.get(
            "final_answer",
            "No answer generated."
        )


        department = result.get(
            "department",
            "unknown"
        )


        confidence = result.get(
            "router_confidence",
            0.0
        )


        sources = result.get(
            "sources",
            []
        )


        is_verified = result.get(
            "is_verified",
            False
        )


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return jsonify({

            "answer": answer,

            "department": department,

            "confidence": f"{confidence:.2%}",

            "sources": sources[:5],

            "verified": is_verified,

            "document_processed": bool(document_path),

            # IMPORTANT:
            # URL of uploaded image/file
            "image_url": image_url,

            # Useful for debugging
            "filename": safe_filename

        })


    except Exception as e:

        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        return jsonify({

            "error": str(e)

        }), 500


# ============================================================
# RESET CONVERSATION
# ============================================================

@app.route("/reset", methods=["POST"])
def reset():

    session_id = session.get("session_id")


    if session_id:

        reset_conversation(
            session_id
        )


    return jsonify({

        "status": "ok"

    })


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )