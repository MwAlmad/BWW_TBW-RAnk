from datetime import datetime

from sqlalchemy import CheckConstraint

from securewallet import db, login_manager
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
@login_manager.user_loader
def load_user(user_id):
    """Check if user is logged-in on every page load."""
    if user_id is not None:
        return User.query.get(user_id)
    return None

class User(db.Model, UserMixin):
    """
create tables
    """
    __tablename__='users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(64), unique=True, index=True)
    password_hash = db.Column(db.String(128),nullable=False)
    account_address = db.Column(db.String, nullable=False)
    privatekey = db.Column(db.String, nullable=False)
    publickey= db.Column(db.String, nullable=False)
    auth = db.Column(db.Integer, nullable=False)
    auth_type = db.Column(db.Integer, nullable=False)
    bio_desc = db.Column(db.Text, nullable=True)
    balance = db.Column(db.Float, nullable=False)

    def __init__(self, email, password, account_address, publickey, privatekey, auth, auth_type, bio_desc,balance):
        self.email = email
        self.password_hash = generate_password_hash(password)
        self.account_address = account_address
        self.publickey = publickey
        self.privatekey = privatekey
        self.auth = auth
        self.auth_type = auth_type
        self.bio_desc = bio_desc
        self.balance = balance

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_keys(self):
        return self.account_address, self.publickey,self.privatekey

    def get_balance(self):
        return self.balance

    def get_email(self):
        return self.email

    def get_auth(self):
        return self.auth

    def get_last_transaction(self):
        user_trans = TransacForUsere.query.filter_by(user_email=self.email).order_by(
            TransacForUsere.transaction_id.desc()).first()
        return user_trans

    def get_all_transactions(self):
        user_trans = TransacForUsere.query.filter_by(user_email=self.email).all()
        return user_trans

    def get_account_address(self,account_address):
        # Retrieve the recipient from the database
        recipient = User.query.filter_by(account_address=account_address).first()
        return recipient

    @classmethod
    def account_address_exists(cls, account_address):
        return cls.query.filter_by(account_address=account_address).first()

    @classmethod
    def get_balance_by_address(cls, address):
        user = cls.query.filter_by(account_address=address).first()
        if user:
            return user.balance
        else:
            return None


    def __repr__(self):
        return f"{self.email}"


class WalletAccount(db.Model):
    # Setup the relationship to the User table
    users = db.relationship(User)
    id = db.Column(db.Integer, primary_key=True)
    # connect the WalletAccount to a particular user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    account_address = db.Column(db.String, nullable=False)
    publickey = db.Column(db.String, nullable=False)
    privatekey = db.Column(db.String, nullable=False)
    balance = db.Column(db.Float, nullable=False)

    def __init__(self, account_address, publickey,privatekey, user_id,balance):
        self.account_address = account_address
        self.publickey = publickey
        self.privatekey = privatekey
        self.user_id = user_id
        self.balance= balance

    def __repr__(self):
        return f"Post Id: {self.id} --- Date: {self.date} --- Publickey: {self.account_address}"


class BWAIRanking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    wallet_name = db.Column(db.String(100), nullable=False)
    support_totp = db.Column(db.String(3), default='No')  # 'Yes' or 'No'
    support_facial_recognition = db.Column(db.String(3), default='No')  # 'Yes' or 'No'
    multiple_cryptocurrencies = db.Column(db.String(3), default='No')  # 'Yes' or 'No'
    wallet_age = db.Column(db.Integer)
    non_custodial = db.Column(db.String(3), default='No')  # 'Yes' or 'No'
    custodial = db.Column(db.String(3), default='No')  # 'Yes' or 'No'
    rating = db.Column(db.Float)  # assuming rating is a decimal value
    security_level = db.Column(db.Integer)  # assuming security level is an integer
    ranking = db.Column(db.Integer)  # overall ranking
    ranking_by_support_totp = db.Column(db.Integer)
    ranking_by_support_facial_recognition = db.Column(db.Integer)
    ranking_by_multiple_cryptocurrencies = db.Column(db.Integer)
    ranking_by_wallet_age = db.Column(db.Integer)
    ranking_by_non_custodial = db.Column(db.Integer)
    ranking_by_custodial = db.Column(db.Integer)
    ranking_by_rating = db.Column(db.Integer)
    ranking_by_security_level = db.Column(db.Integer)

    @staticmethod
    def bool_to_string(value):
        return 'Yes' if value else 'No'

    @staticmethod
    def string_to_bool(value):
        return value == 'Yes'


    def __init__(self, wallet_name, support_totp, support_facial_recognition,
                 multiple_cryptocurrencies, wallet_age, non_custodial,
                 custodial, rating, security_level):
        self.wallet_name = wallet_name
        self.support_totp = support_totp
        self.support_facial_recognition = support_facial_recognition
        self.multiple_cryptocurrencies = multiple_cryptocurrencies
        self.wallet_age = wallet_age
        self.non_custodial = non_custodial
        self.custodial = custodial
        self.rating = rating
        self.security_level = security_level

    def update_rankings(self,ranking=None, totp_ranking=None, facial_recognition_ranking=None,
                        cryptocurrencies_ranking=None, age_ranking=None,
                        non_custodial_ranking=None, custodial_ranking=None,
                        rating_ranking=None, security_level_ranking=None):
        # Update the rankings only if they are provided
        if ranking is not None:
            self.ranking = ranking
        if totp_ranking is not None:
            self.ranking_by_support_totp = totp_ranking
        if facial_recognition_ranking is not None:
            self.ranking_by_support_facial_recognition = facial_recognition_ranking
        if cryptocurrencies_ranking is not None:
            self.ranking_by_multiple_cryptocurrencies = cryptocurrencies_ranking
        if age_ranking is not None:
            self.ranking_by_wallet_age = age_ranking
        if non_custodial_ranking is not None:
            self.ranking_by_non_custodial = non_custodial_ranking
        if custodial_ranking is not None:
            self.ranking_by_custodial = custodial_ranking
        if rating_ranking is not None:
            self.ranking_by_rating = rating_ranking
        if security_level_ranking is not None:
            self.ranking_by_security_level = security_level_ranking

    def get_rankings_by_feature(selected_feature):
        # Mapping of form options to database columns
        feature_to_column_mapping = {
            'General Ranking': BWAIRanking.ranking,
            'Support TOTP': BWAIRanking.support_totp,
            'Support Facial Recognition': BWAIRanking.support_facial_recognition,
            'Multiple Cryptocurrencies': BWAIRanking.multiple_cryptocurrencies,
            'Wallet Age': BWAIRanking.wallet_age,
            'Non-Custodial': BWAIRanking.non_custodial,
            'Custodial': BWAIRanking.custodial,
            'Rating': BWAIRanking.rating,
            'Security Level': BWAIRanking.security_level

        }

        # Get the corresponding database column for the selected feature
        column = feature_to_column_mapping.get(selected_feature)

        # If the selected feature is valid, query the database
        if column:
            rankings = BWAIRanking.query.order_by(column.desc()).all()  # Adjust the ordering as needed
            return rankings

        # Return None or an empty list if the feature is not found
        return None


class TransacForUsere(db.Model):
    __tablename__ = 'transacforusere'
    # Setup the relationship to the User table
    user = db.relationship('User', backref=db.backref('transacforusere', lazy=True))
    transaction_id = db.Column(db.String, primary_key=True)
    block_id = db.Column(db.Integer, db.ForeignKey('block.block_index'), nullable=True)
    # connect a transcript to a particular user
    user_email = db.Column(db.String(64), db.ForeignKey('users.email'), nullable=False)
    date = db.Column(db.String(64), nullable=False)
    sender_address = db.Column(db.String(140), nullable=False)
    recipient_address = db.Column(db.String(140), nullable=False)
    value = db.Column(db.Float, nullable=False)
    signature = db.Column(db.String(128), nullable=False)

    def __init__(self, transaction_id,user_email,sender_address, recipient_address, value,signature, date=datetime.now().strftime("%Y-%m-%d %H:%M")):
        self.transaction_id=transaction_id
        self.user_email = user_email
        self.sender_address = sender_address
        self.recipient_address = recipient_address
        self.value = value
        self.signature=signature
        self.date = date

    @property
    def get_transaction(self):
        return self.sender_address, self.recipient_address, self.value


    def __repr__(self):
        return f"Transaction Id: {self.transaction_id} --- Date: {self.date}  --- Sender_address: {self.sender_address} --- Recipient_address: {self.recipient_address} ---Value: {self.value}"

class Block(db.Model):
    __tablename__ = 'block'

    block_index = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    nonce = db.Column(db.String(140), nullable=False)
    previous_hash = db.Column(db.String(140), db.ForeignKey('block.block_hash'), nullable=False)
    block_hash = db.Column(db.String(140),)

    __table_args__ = (
        CheckConstraint('block_index > 0', name='positive_block_index'),
    )

    def __init__(self, block_number, previous_hash):
        self.block_number = block_number
        self.previous_hash = previous_hash

    def __repr__(self):
        return f"Post Id: {self.block_number} --- Date: {self.date} --- previous_hash: {self.previous_hash}"


class AddNode(db.Model):
    node_id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    nodes_url = db.Column(db.String(500), unique=True, nullable=False)

    def __init__(self, nodes_url):
        self.nodes_url = nodes_url
    def __repr__(self):
        return f"Post Id: {self.node_id} --- Date: {self.date} --- node_url: {self.nodes_url}"











