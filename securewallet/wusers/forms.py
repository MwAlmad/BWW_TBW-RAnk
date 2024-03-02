from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, RadioField, validators,FloatField,TextAreaField
from wtforms.fields.simple import HiddenField
from wtforms.validators import DataRequired,Email,EqualTo, Length
from wtforms import ValidationError
from securewallet.models import User
from securewallet.generate_wallet.wallets import Waccount

account = Waccount ()

class LoginForm(FlaskForm):
    email = StringField('Email',validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Log In')


class SecsettingForm(FlaskForm):
    otp_url = HiddenField()
    auth_type = HiddenField()
    bio_desc = HiddenField()
    auth_type_radio = RadioField('Label1',[validators.DataRequired()],
                                 choices=[('0', '(S1) Passwords Only'),
                                          ('1', '(S2) Passwords + TOTP'),
                                          ('2', '(S3) Passwords + Face Recognition'),
                                          ('3', '(S4) Passwords + TOTP + Face Recognition')], default='0')
    submit = SubmitField('Save')
    # email = StringField('Email', validators=[DataRequired(), Email()])

class RegistrationForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(),Email()])
    password = PasswordField('Password', validators=[DataRequired(),EqualTo('pass_confirm', message='Passwords Must Match!'),Length(min=8, message="Password must be at least 8 characters")])
    pass_confirm = PasswordField('Confirm password', validators=[DataRequired(),EqualTo('password', message='Passwords Must Match!')])
    submit = SubmitField('Signup')
    auth = RadioField('Label1',
                      [validators.DataRequired()],
                      choices=[('value', 'Primary Authentication'),
                               ('value_two', 'Two-Factor Authentication'),
                               ('value_three', 'Biometric Authentication')], default='value')

    acc_add = account.account_address
    pub_key = account.publickey
    pri_key = account.privatekey
    balance = account.get_balance()


    def validate_email(self, email):
        # Check if not None for that user email!
        if User.query.filter_by(email=email.data).first():
            raise ValidationError('Your email has been registered already!')



class OTPForm(FlaskForm):
    otp = PasswordField('TOTP', validators=[DataRequired()])
    submit = SubmitField('Submit')

class FaceAuthForm(FlaskForm):
    password = PasswordField('Biometric Authentication')
    submit = SubmitField('Login')
    # bio_desc = HiddenField()


