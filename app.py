from flask import Flask, render_template, url_for, redirect, request
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import Required

from flask_bootstrap import Bootstrap

SECRET_KEY = 'some secret key'

app = Flask(__name__)
app.config.from_object(__name__)
bootstrap = Bootstrap(app)

BASE_DIR = './data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'

import glob
import random
import codecs

def getRandomEmail():
	if random.random() >= 0.5:
		filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
	else:
		filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
	n = random.randint(0, len(filenames))
	filename = filenames[n]
	with codecs.open(filename, encoding='latin-1') as f:
		return f.read()

class TextForm(FlaskForm):
	text = TextAreaField('Email', validators=[Required()], render_kw={"placeholder": "Input your own email or generate a random email."})
	submit = SubmitField('Classify')

import spam_classifier as spam

X_train, y_train = spam.load_data()
model, loss = spam.train_gd(X_train, y_train, 0.4, 0.1)

@app.route('/', methods=['GET', 'POST'])
def index():
	res = None
	email = ''
	form = TextForm()
	if request.method == 'POST' and request.form['submit'] == 'Generate Random Email':
		form.text.data = getRandomEmail()
	elif request.method == 'POST':
		if form.validate_on_submit():
			email = form.text.data
			inputText = spam.generate_feature_vector(email)
			res = spam.predict(inputText.dot(model), 0.9)[0]
	return render_template('index.html', form=form, res=res, email=email)

if __name__ == '__main__':
	app.run(debug=True)