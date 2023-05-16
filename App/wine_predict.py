from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

print(os.getcwd())

# 피클 파일에서 훈련된 모델을 로드
# 노트북에서 경로가 안잡혀서 절대경로로 해 놓았습니다. 
with open("C:\\Users\\hyj73\\OneDrive\\문서\\GitHub\\RedWind_Model\\RedWine.pkl","rb") as f:
    model = pickle.load(f)

@app.route('/main')
def main():
    return render_template('main.html')

@ app.route ( "/main/predict", methods = ["POST"])
def predict():
    input_data ={
        "fixed_acidity": request.form['fixed_acidity'],
        "volatile_acidity": request.form['volatile_acidity'],
        "citric_acid": request.form['citric_acid'],
        "residual_sugar": request.form['residual_sugar'],
        "chlorides": request.form['chlorides'],
        "free_sulfur_dioxide": request.form['free_sulfur_dioxide'],
        "total_sulfur_dioxide": request.form['total_sulfur_dioxide'],
        "density": request.form['density'],
        "pH": request.form['pH'],
        "sulphates": request.form['sulphates'],
        "alcohol": request.form['alcohol']
    }

# json으로 변환
    input_json = json.dumps(input_data)
    input_data_as_numpy_array = np.asarray(list(input_data.values()))
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # 예측 수행
    prediction = model.predict(input_data_reshaped)

    if prediction[0] == 1:
        print("Good Quality Wine!")
        result = "Good Quality"
    else:
        print("Bad Quality Wine")
        result = "Bad Quality"

    #웹 페이지 리턴
    return render_template('result.html', result=result)



if __name__ == '__main__':
    app.run(debug=True)