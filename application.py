from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/',methods=['GET','POST'])
#def home_page():
#    return render_template('index.html')

#@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        data=CustomData(
            age=float(request.form.get('age')),
            sex = float(request.form.get('sex')),
            on_thyroxine = float(request.form.get('on_thyroxine')),
            query_on_thyroxine = float(request.form.get('query_on_thyroxine')),
            on_antithyroid_medication = float(request.form.get('on_antithyroid_medication')),
            
            sick = float(request.form.get('sick')),
            pregnant = float(request.form.get('pregnant')),
            thyroid_surgery = float(request.form.get('thyroid_surgery')),
            I131_treatment = float(request.form.get('I131_treatment')),
            query_hypothyroid = float(request.form.get('query_hypothyroid')),
            query_hyperthyroid = float(request.form.get('query_hyperthyroid')),
            lithium = float(request.form.get('lithium')),
            goitre = float(request.form.get('goitre')),
            tumor = float(request.form.get('tumor')),

            hypopituitary = float(request.form.get('hypopituitary')),
            psych = float(request.form.get('psych')),
            TSH_measured = float(request.form.get('TSH_measured')),
            TSH = float(request.form.get('TSH')),
            T3_measured = float(request.form.get('T3_measured')),
            T3 = float(request.form.get('T3')),
            TT4_measured = float(request.form.get('TT4_measured')),
            TT4 = float(request.form.get('TT4')),
            T4U_measured = float(request.form.get('T4U_measured')),
            T4U = float(request.form.get('T4U')),
            FTI_measured = float(request.form.get('FTI_measured')),
            FTI = float(request.form.get('FTI'))
            
           
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        if pred[0] == 0:
            results="YES"
        else:
            results="NO"
        #results=round(pred[0],2)

        return render_template('results.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True, port=5005)
##debug=True
