from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField,StringField
from wtforms.validators import DataRequired

class FeatureSelectionForm(FlaskForm):

    feature_choices = [
        ('0', 'General Ranking'),
        ('1', 'Support TOTP'),
        ('2', 'Support Facial Recognition'),
        ('3', 'Multiple Cryptocurrencies'),
        ('4', 'Wallet Age'),
        ('5', 'Non-Custodial'),
        ('6', 'Custodial'),
        ('7', 'Rating'),
        ('8', 'Security Level'),
    ]
    selected_feature = RadioField('Select a Feature', choices=feature_choices, validators=[DataRequired()], default='0')
    Wallet_Name = StringField('Wallet Name', validators=[DataRequired()])
    submit = SubmitField('Show Rankings')