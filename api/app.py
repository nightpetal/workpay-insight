from flask import Flask, render_template, request
import pandas as pd
import joblib, os
from google import genai
import key

app = Flask(__name__, template_folder="../templates", static_folder="../static")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

saved = joblib.load(MODEL_PATH)
models = saved["models"]
columns = saved["columns"]

# Store ai chat history in memory (reset on server restart)
chat_history = [{"sender": "Bot", "message": "Ask me about your questions"}]
instructions = "no markdown, based on data, how can user improve or get promotion"


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None

    if request.method == "POST":

        form_type = request.form.get("form_type")

        if form_type == "prediction":
            input_data = {
                "Experience_Level_encoded": int(request.form["experience"]),
                "Project_Type_encoded": int(request.form["project_type"]),
                "Job_Category": request.form["job_category"],
                "Platform": request.form["platform"],
                "Client_Region": request.form["client_region"],
                "Hours_Worked_Per_Week": float(request.form["hours"]),
            }

            df = pd.DataFrame([input_data])
            df = pd.get_dummies(df)
            df = df.reindex(columns=columns, fill_value=0)

            predictions = {
                target: round(models[target].predict(df)[0], 2) for target in models
            }
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=f"{instructions} What does the user need to improve their salary based on this: {input_data}",
                )
                print(response.text)
                assistant_msg = getattr(response, "text", "AI response empty")
                print(assistant_msg)

            except Exception as e:
                assistant_msg = (
                    f"AI is unavailable right now. Please try again later. ({e})"
                )
            chat_history.append({"sender": "Bot", "message": assistant_msg})

        elif form_type == "chat":
            user_msg = request.form.get("chat_input", "").strip()
            if user_msg:
                chat_history.append({"sender": "User", "message": user_msg})

                try:
                    client = genai.Client(api_key=GEMINI_API_KEY)
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=f"{instructions}old history{chat_history}request{user_msg}",
                    )
                    print(response.text)
                    assistant_msg = getattr(response, "text", "AI response empty")
                    print(assistant_msg)

                except Exception as e:
                    assistant_msg = (
                        f"AI is unavailable right now. Please try again later. ({e})"
                    )

                chat_history.append({"sender": "Bot", "message": assistant_msg})

                print(chat_history)

    return render_template(
        "index.html", predictions=predictions, chat_history=chat_history
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
