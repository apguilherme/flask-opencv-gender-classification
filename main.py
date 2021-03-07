from flask import Flask
import views

app = Flask(__name__)

app.add_url_rule("/base", "base", views.base, methods=["GET"])
app.add_url_rule("/", "index", views.index, methods=["GET", "POST"])

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

