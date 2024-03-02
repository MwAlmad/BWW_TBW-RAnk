from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, DateField, FloatField, FieldList, FormField, HiddenField
from wtforms.validators import DataRequired

class AddNodeForm(FlaskForm):
    nodes_url = StringField('Node_url', validators=[DataRequired()])
    submit = SubmitField('Add a Node')

class GenerateTransactionForm (FlaskForm):

    sender_address = StringField('Sender_address', validators=[DataRequired()])
    sender_private_key =StringField('Sender_private_key', validators=[DataRequired()] )
    recipient_address = StringField('Recipient_address', validators=[DataRequired()])
    value = FloatField('Amount', validators=[DataRequired()])
    submit = SubmitField('Generate Transaction')

class BlockForm(FlaskForm):

    block_index = IntegerField()
    date = DateField()
    transactions = StringField()
    nonce = IntegerField()
    previous_hash = StringField()
    submit = SubmitField('Mine Transactions')

