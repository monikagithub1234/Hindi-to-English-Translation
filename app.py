from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if text == "":
            return jsonify({"english": ""})

        # Hindi input
        tokenizer.src_lang = "hin_Deva"
        inputs = tokenizer(text, return_tensors="pt")

        # English token (SAFE METHOD)
        eng_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=eng_token_id,
            max_length=128,
            num_beams=5
        )

        translated_text = tokenizer.decode(
            translated_tokens[0],
            skip_special_tokens=True
        )

        return jsonify({"english": translated_text})

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"english": "Translation failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)
