from flask import Flask, render_template, url_for, redirect
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import Required

from flask_bootstrap import Bootstrap

SECRET_KEY = 'some secret key'

app = Flask(__name__)
app.config.from_object(__name__)
bootstrap = Bootstrap(app)

class TextForm(FlaskForm):
	text = TextAreaField('Email', validators=[Required()])
	submit = SubmitField('Classify')

import spam_classifier as spam

X_train, y_train = spam.load_data()
model, loss = spam.train_gd(X_train, y_train, 0.4, 0.1)

@app.route('/', methods=['GET', 'POST'])
def index():
	form = TextForm()
	res = None
	email = ''
	if form.validate_on_submit():
		inputText = spam.generate_feature_vector(form.text.data)
		res = spam.predict(inputText.dot(model), 0.9)[0]
		email = form.text.data
	return render_template('index.html', form=form, res=res, email=email)

if __name__ == '__main__':
	app.run(debug=False)