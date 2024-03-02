from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FloatField,validators
from wtforms.validators import DataRequired

class GenerateWalletForm(FlaskForm):


    account_address = StringField('Account_Address', validators=[DataRequired()])
    privatekey = StringField('Private Key', validators=[DataRequired()])
    publickey = StringField('Public Key', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    balance = FloatField ('Your balance', validators=[DataRequired()])
    submit = SubmitField('Generate Wallet', validators=[DataRequired()])