from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('Loan_R_F_model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_placement():
    age = int(request.form.get('age'))
    income = int(request.form.get('income'))
    credit_score = int(request.form.get('credit_score'))

    #Encoding categorical values (Modify based on your model's training)
    #gender = 1 if gender.lower() == 'male' else 0  # Example encoding

    # prediction
    result = model.predict(np.array([age,income,credit_score],dtype= int).reshape(1,3))

    if result[0] == 0:
        result = 'approved'
    else:
        result = 'denied'

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)









