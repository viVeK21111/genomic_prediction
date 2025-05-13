import pandas as pd
import numpy as np
from joblib import load
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress specific warnings from scikit-learn
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask,redirect,url_for,render_template,request,send_from_directory
from flask_compress import Compress

#scaler = load('model/scaler_x.pkl')
lmc=load('model/model_r.pkl')
lmr=load('model/model_br.pkl')  
# lmc=load_model('testc.h5') deep learning model(ann) for classification



## flask 


app=Flask(__name__)
Compress(app)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Home')
def Home():
    return render_template('home.html')

@app.route('/pred')
def pred():
    return render_template('index.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/phylo')
def phylo(): 
    return render_template('datat.html')

@app.route('/about')
def about():
    return render_template('about.html') 

@app.route('/func2') 
def func2():
    return render_template('wlen.html') 

@app.route('/empty')
def empty():
    return render_template('empty.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory('download',filename,as_attachment=True)

@app.route('/inv')
def inv():
    return "<h2>Invalid input</h2>"

@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        import re
        seq=str(request.form['seq']) 
        if(len(seq)==0):
            return redirect(url_for('empty'))
        alc='ATCG'
        SEQ=re.findall(r'\b\w+\b',seq)  
        sel=str(SEQ)
        sel=sel.replace('[',"")
        sel=sel.replace(']',"")
        sel=sel.replace("'","")
        sel=sel.replace(" ","") 
        sel=list(sel)
        sel=set(sel)
        if(len(SEQ)>1): 
            for i in sel:
                if(i!='A' and i!='T' and i!='C' and i!='G' and i!=','):
                    return redirect(url_for('inv'))
        else:
            for i in sel:
                if(i!='A' and i!='T' and i!='C' and i!='G'):
                    return redirect(url_for('inv'))
        l=len(SEQ)
        flag=0
        p=0
        for i in range(l):
            if(len(SEQ[i])==9): 
                flag=0
            else:
                flag=1
                p=i
                break
        if(flag==0):
            ss=SEQ 
            nuc=['A','T','C','G']
            dicc={'A':1.5,'T':2.5,'C':0.5,'G':0.75}
            encoded=[]
            for i in ss:
                array = np.array(list(i))
                new_arr=np.array([dicc[i] for i in array])
                encoded.append((new_arr))

            enc_dfff=pd.DataFrame(encoded)
            zz=lmc.predict(enc_dfff)  
            id=list(zz)
            enc_dfff['sub']=id
            enc_dfff.columns=enc_dfff.columns.astype(str)
            enc_dfff['sub']=enc_dfff['sub'].astype('category').cat.codes +1 
            #scaler.fit(enc_dfff)
            #enc_dfff= scaler.transform(enc_dfff)
            enc_dfff=enc_dfff.values
            res=lmr.predict(enc_dfff)
            res_df=pd.DataFrame()   
            res_df['sequence']=ss
            res_df['subpopulation']=id 
            res_df['height']=res
            rl=res_df.values.tolist() 
            return render_template("dataf.html",rl=rl) 
        else:
            return redirect(url_for('func2'))   
    else:
        return "invalid request" 
if __name__=="__main__":
    app.run(threaded=True) 
