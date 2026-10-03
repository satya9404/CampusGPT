from flask import Flask, render_template, request, jsonify

from rag.generator import generate_answer


app = Flask(__name__)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# ASK CAMPUSGPT
# ============================================================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No question received."
            })


        question = data.get(
            "question",
            ""
        ).strip()


        if not question:

            return jsonify({
                "success": False,
                "error": "Please enter a question."
            })


        # ----------------------------------------------------
        # Generate CampusGPT answer
        # ----------------------------------------------------

        result = generate_answer(
            question,
            top_k=5
        )


        return jsonify({

            "success": True,

            "answer": result.get(
                "answer",
                "I couldn't generate an answer."
            ),

            "sources": result.get(
                "sources",
                []
            )

        })


    except Exception as error:

        print(
            "APP ERROR:",
            error
        )

        return jsonify({

            "success": False,

            "error": (
                "CampusGPT encountered an "
                "unexpected error."
            )

        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )