import os
import re
from flask import Flask, render_template, request, redirect, url_for, abort,send_from_directory
from flask.helpers import send_file
from werkzeug.utils import secure_filename
import pandas as pd
import csv
from utils import empty_contents, get_categorical_missing_attributes, missing_value_graph,get_numeric_attributes,create_histogram,create_pairplot,create_scatterplot,create_barplot,get_attributes,set_var
from utils import get_categorical,get_numeric,get_target,get_unwanted,get_attributes,get_numeric_missing_attributes,update_csv_,empty_contents,encode_categorical,train_model


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5120 * 5120
app.config['UPLOAD_EXTENSIONS'] = ['.csv']
app.config['UPLOAD_PATH'] = 'uploads/'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300
FILEPATH = 'uploads/'
DOWNLOAD_PATH = ''
counter=-1
counter2=0
temp=''
ctr=0
attr_ui=[]

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.cache_control.max_age = 300
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/filter-attributes',methods=['GET','POST'])
def filter_attr():
    if request.method=='GET':
        attr=get_attributes(FILEPATH)
        return render_template('filter_attributes.html',context={'attr':attr})
    elif request.method=='POST':
        cat_var = request.form.getlist('choice1')
        tar_var = request.form.getlist('choice2')
        num_var = request.form.getlist('choice3')
        un_var  = request.form.getlist('choice4')
        set_var(cat_var,tar_var,num_var,un_var)
        print(cat_var,tar_var,num_var,un_var)
        return redirect(url_for('preprocess')) 


@app.route('/upload', methods=['GET','POST'])
def upload_files():
    if request.method == 'POST':
        global FILEPATH
        global counter
        global counter2
        global attr_ui
        global temp,ctr
        if ctr>0:
            os.remove(temp)
        ctr+=1
        FILEPATH ='uploads/'
        uploaded_file = request.files['userfile']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']: 
                abort(400)
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
            FILEPATH += filename
            temp=FILEPATH
            dataframe = pd.read_csv('uploads/'+filename)
            attr_names = dataframe.columns
            lst = []
            for i in range(dataframe.shape[0]):
                sub_lst=[]
                for j in range(dataframe.shape[1]):
                    sub_lst.append(dataframe.iloc[i][j])
                lst.append(sub_lst)
        context = {'filename':filename,'lst':lst,'attr':attr_names}
        empty_contents()
        counter=-1
        counter2=0
        attr_ui=[]
        print("Counter={},Counter2={},attr_ui={}".format(counter,counter2,attr_ui))
        return render_template('upload.html',context=context)
    return render_template('upload.html',context={})

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

@app.route('/visualize/histogram',methods=['GET','POST'])
def histogram_vis():
    if request.method=='GET':
        num_attr = get_numeric()
        return render_template('histogram.html',context={'num_attr':num_attr,'id':2})
    elif request.method=="POST":
        attr= request.form.get("attribute")
        bins = request.form.get("bins")
        file = create_histogram(attr,bins,FILEPATH)
        return render_template('histogram.html',context={'id':1,'file':file,'attr':attr})
    

@app.route('/visualize/barplots',methods=['GET','POST'])
def barplot_vis():
    if request.method=='GET':
        cat_attr = get_categorical()
        num_attr = get_numeric()
        return render_template('barplot.html',context={'num_attr':num_attr,'cat_attr':cat_attr,'id':2})
    elif request.method=="POST":
        attr1 = request.form.get("attribute1")
        attr2 = request.form.get("attribute2")
        file = create_barplot(attr1,attr2,FILEPATH)
        return render_template('barplot.html',context={'id':1,'file':file,'attr1':attr1,'attr2':attr2})
    

@app.route('/visualize/scatterplot',methods=['GET','POST'])
def scatterplot_vis():
    if request.method=='GET':
        num_attr = get_numeric()
        return render_template('scatter.html',context={'num_attr':num_attr,'id':2})
    elif request.method=="POST":
        attr1 = request.form.get("attribute1")
        attr2 = request.form.get("attribute2")
        file = create_scatterplot(attr1,attr2,FILEPATH)
        return render_template('scatter.html',context={'id':1,'file':file,'attr1':attr1,'attr2':attr2})
    
    

@app.route('/visualize/pairplot')
def pairplots_vis():
    file=create_pairplot(FILEPATH)
    return render_template('pairplot.html',context={'file':file})


@app.route('/preprocess')
def preprocess():
    return render_template('preprocess.html',context={'id':0})

@app.route('/preprocess_',methods=['GET','POST'])
def update_csv():
    global attr_ui
    if request.method=='POST':
        print("Inside Update.app")
        print("update_csv=",attr_ui)
        update_csv_(attr_ui,FILEPATH)
        return render_template('preprocess.html',context={'id':1})


@app.route('/preprocess/missing-data-handling',methods=['GET','POST'])
def missing_pre():
    print("In Here")
    global counter
    global counter2
    global attr_ui
    print("Counter=",counter)
    num_miss = get_numeric_missing_attributes(FILEPATH)
    print(num_miss)
    cat_miss = get_categorical_missing_attributes(FILEPATH)
    if counter==-1:
        counter+=1
        print("vapas idhar")
        file = missing_value_graph(FILEPATH)
        return render_template('missing-data-handling.html',context={'file':file,'id':1})
    if request.method=="POST":
        if num_miss:
            if counter==0:
                counter+=1
                return render_template('missing-data-handling.html',context={'id':2,'num_miss':num_miss[counter-1]})
            elif counter>0 and counter<len(num_miss):
                a = num_miss[counter-1]
                b = request.form.get('constant')
                c = request.form.get('rg')
                attr_ui.append((a,b,c))
                print(attr_ui)
                counter+=1
                return render_template('missing-data-handling.html',context={'id':2,'num_miss':num_miss[counter-1]})
            elif counter==len(num_miss):
                a=num_miss[counter-1]
                b=request.form.get('constant')
                c=request.form.get('rg')
                attr_ui.append((a,b,c))
                print(attr_ui)
                counter+=1
                #counter=-1
                counter2+=1
                if cat_miss:
                    return render_template('missing-data-handling.html',context={'id':3,'cat_miss':cat_miss[counter2-1]})
                return render_template('missing-data-handling.html',context={'id':4})
            elif counter>len(num_miss) and (counter2-1)<len(cat_miss):
                a=cat_miss[counter2-1]
                b=''
                c=request.form.get('rg0')
                attr_ui.append((a,b,c))
                print(attr_ui)
                counter2+=1
                if counter2>len(cat_miss):#== instad of >
                    a=cat_miss[-1]
                    b=''
                    c=request.form.get('rg0')
                    attr_ui.append((a,b,c))
                    print(attr_ui)
                    counter2=-1
                    #counter2=0
                    return render_template('missing-data-handling.html',context={'id':4})
                return render_template('missing-data-handling.html',context={'id':3,'cat_miss':cat_miss[counter2-1]})
        elif cat_miss:
            if counter==len(num_miss):
                counter2+=1
                counter+=1
                return render_template('missing-data-handling.html',context={'id':3,'cat_miss':cat_miss[counter2-1]})
            elif (counter2-1)<len(cat_miss):
                a=cat_miss[counter2-1]
                b=''
                c=request.form.get('rg0')
                attr_ui.append((a,b,c))
                print(attr_ui)
                counter2+=1
                if counter2>len(cat_miss):#== instad of >
                    a=cat_miss[-1]
                    b=''
                    c=request.form.get('rg0')
                    attr_ui.append((a,b,c))
                    print(attr_ui)
                    counter2=-1
                    #counter2=0
                    return render_template('missing-data-handling.html',context={'id':4})
                return render_template('missing-data-handling.html',context={'id':3,'cat_miss':cat_miss[counter2-1]})
        else:
            return redirect(url_for('base.html'))
    

        
@app.route('/preprocess/encode-categorical',methods=['GET','POST'])
def encode_pre():
    if request.method=='POST':
        enc = request.form.get('enc')
        encode_categorical(enc,FILEPATH)
        return redirect(url_for('visualize'))
    return render_template('encode-categorical.html')

@app.route('/train_model/choose-algorithm',methods=['GET','POST'])
def choose_algorithm():
    global DOWNLOAD_PATH
    if request.method=='POST':
        type = request.form.get('type')
        test_size = request.form.get('ts')
        print(test_size)
        if type=='reg':
            algo = request.form.get('ch1')
        elif type=='class':
            algo = request.form.get('ch2')
        lst,flag,model_path = train_model(FILEPATH,algo,test_size)
        DOWNLOAD_PATH = model_path
        if flag=='c':
            return render_template('model_performance.html',context={'flag':flag,'lst':lst,'path':model_path})
        elif flag=='r':
            return render_template('model_performance.html',context={'flag':flag,'lst':lst,'path':model_path})
        
    return render_template('choose-algorithm.html')

@app.route('/download_model')
def download():
    global DOWNLOAD_PATH
    return send_file(DOWNLOAD_PATH,as_attachment=True)

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)