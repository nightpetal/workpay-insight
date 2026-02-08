# WorkPay Insight

**WorkPay Insight** is a Flask web application that predicts freelancer earnings per hour, estimates possible client ratings, and provides a chatbot interface using Gemini for queries. The project uses a locally trained machine learning model and is deployment-ready for services like Render.

---

## **Features**

- Predict hourly earnings of freelancers based on user input.
- Estimate potential client ratings for projects.
- Interactive chatbot powered by Gemini API for user queries.
- Uses locally trained ML model stored in `model/model.pkl`.
- Secure handling of API keys via environment variables.

---

## **Project Structure**

```
workpayinsight/
├── api/
│   └── app.py
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── model/
│   ├── freelancer_earnings_bd.csv
│   ├── model_randomforest.py
│   └── model.pkl
├── requirements.txt
├── LICENSE
└── README.md
```

---

## **Setup (Local Development)**

1. Clone the repository:

```bash
git clone https://github.com/nightpetal/workpay-insight.git
cd workpay-insight
```

2. Create a virtual environment and activate it:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set environment variables locally (optional `.env` file):

```
GEMINI_API_KEY=your_api_key_here
```

5. Run the app locally:

```bash
flask --app api.app run
```

---

## **Deployment (Render)**

1. Ensure `requirements.txt` includes Gunicorn:

```
Flask
joblib
python-dotenv
gunicorn
```

2. Set the start command in Render:

```
gunicorn api.app:app
```

3. Add environment variables in Render dashboard:

```
GEMINI_API_KEY = your_api_key_here
```

4. Deploy the web service. Render will automatically install dependencies and start the Flask app.

---

## **Usage**

- Open the app in your browser (Render URL or localhost).
- Fill in the freelancer/project details.
- Receive predicted hourly earnings and client rating.
- Interact with the Gemini chatbot for additional queries.

---

## **Contributing**

- Clone the repository and create a feature branch:

```bash
git checkout -b feature/new-feature
```

- Commit your changes and push:

```bash
git add .
git commit -m "Describe your changes"
git push origin feature/new-feature
```

- Open a pull request on GitHub.

---

## **License**

MIT License – see [MIT License](LICENSE) file.

---
