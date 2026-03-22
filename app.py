from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os

app = Flask(__name__)

# ---------- LOAD DATA ----------
data = pd.read_csv("student_data.csv")

X = data.drop("final_score", axis=1)
y = data["final_score"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# ---------- ROUTES ----------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 👇 Student Info
        name = request.form["name"]
        usn = request.form["usn"]
        branch = request.form["branch"]

        # 👇 Academic Inputs
        prev_cgpa = float(request.form["prev_cgpa"])
        prev_sgpa = float(request.form["prev_sgpa"])
        curr_cgpa = float(request.form["curr_cgpa"])
        curr_sgpa = float(request.form["curr_sgpa"])
        attendance = float(request.form["attendance"])
        project = float(request.form["project"])

        input_data = pd.DataFrame([{
            "prev_cgpa": prev_cgpa,
            "prev_sgpa": prev_sgpa,
            "curr_cgpa": curr_cgpa,
            "curr_sgpa": curr_sgpa,
            "attendance": attendance,
            "project": project
        }])

        prediction = model.predict(input_data)[0]

        # ---------- GRAPH ----------
        labels = ["Prev CGPA", "Curr CGPA", "Attendance", "Project", "Predicted"]
        values = [prev_cgpa, curr_cgpa, attendance/10, project/10, prediction/10]

        plt.figure()
        plt.bar(labels, values)

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        graph_url = base64.b64encode(img.getvalue()).decode()

        return render_template(
            "result.html",
            prediction=round(prediction, 2),
            graph_url=graph_url,
            name=name,
            usn=usn,
            branch=branch
        )

    except Exception as e:
        return str(e)


# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)