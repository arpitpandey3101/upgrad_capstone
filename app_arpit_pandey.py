# -*- coding: utf-8 -*-

from flask import Flask,render_template,request
import model 
app = Flask('__name__')

valid_users = ['00sab00','1234','zippy','zburt5','joshua','dorothy w','rebecca','walker557','samantha','raeanne','kimmie','cassie','moore222']
@app.route('/')
def view_screen():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommendation_best5():
    print(request.method)
    userName = request.form['User Name']
    print('User Name=',userName)
    
    if  userName in valid_users and request.method == 'POST':
            products_best20 = model.recommend_products(userName)
            print(products_best20.head())
            best5_get = model.top5_products(products_best20)
            return render_template('index.html',column_names=best5_get.columns.values, row_data=list(best5_get.values.tolist()), zip=zip,text='Product Recommendations')
    elif not userName in  valid_users:
        return render_template('index.html',text='Not enough data or invalid user')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.debug=False

    app.run()