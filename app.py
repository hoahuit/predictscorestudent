from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Tải mô hình và dữ liệu
model = joblib.load('student_score_model.pkl')
data = pd.read_csv("data.csv")  # Thay bằng tệp dữ liệu thực tế của bạn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_schools', methods=['GET'])
def get_schools():
    # Lấy danh sách trường học duy nhất từ dữ liệu
    schools = data['school'].dropna().unique().tolist()
    return jsonify({"schools": schools})
@app.route('/predict', methods=['POST'])
def predict():
    data_input = request.json
    # Chuyển đổi dữ liệu đầu vào thành DataFrame
    new_student = pd.DataFrame([data_input])
    # Dự đoán điểm số
    predicted_score = model.predict(new_student)[0]
    return jsonify({"predicted_pretest_score": predicted_score})

if __name__ == '__main__':
    app.run(debug=True)
