import codecs
import hashlib
import uuid
from flask import render_template, url_for, flash, redirect, request, Blueprint, session, Response
from flask_login import login_manager, login_user, current_user, logout_user, login_required
from sqlalchemy import update
from werkzeug.wrappers import Response
from securewallet import db
from securewallet.models import User,Block,TransacForUsere
from securewallet.wusers.forms import RegistrationForm, LoginForm, SecsettingForm, OTPForm, FaceAuthForm
from securewallet.generate_wallet.views import GenerateWalletForm
from securewallet.bc_implementation.forms import GenerateTransactionForm
from securewallet.generate_wallet.wallets import Transaction, InvalidTransactionException, InsufficientFundsException
import pyotp
import base64


users = Blueprint('users', __name__)

@users.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration route."""
    if current_user.is_authenticated:
        return redirect(url_for('core.index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if user is not None:
            flash('Email already exists, Please sign in.', category='error')
            return redirect(url_for('login'))
        user = User(email=form.email.data, password=form.password.data, account_address=form.acc_add,publickey=form.pub_key, privatekey=form.pri_key, auth=0, auth_type=0, bio_desc=None,balance=form.balance)
        db.session.add(user)
        db.session.commit()
        flash('Thanks for registering! Now you can Sign in!', category='success')

        session['email'] = user.email
        return redirect(url_for('users.login'))

    return render_template('signup.html', form=form)


@users.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        # if user is logged in we get out of here
        return redirect(url_for('core.index'))
    form = LoginForm()
    if form.validate_on_submit():
        # Grab the user from our User Models table
        user = User.query.filter_by(email=form.email.data).first()
        if user is None or not user.check_password(form.password.data) :
            flash('Invalid email, password.', category='error')
            return redirect(url_for('users.login'))

        if user.auth_type == 0:
            login_user(user)
            flash('Logged in successfully.', category='success')
            return redirect(url_for('core.index'))

        elif user.auth_type == 1:
            return redirect(url_for('users.otp', email=form.email.data))
        elif user.auth_type == 2:
            bio_desc = user.bio_desc
            return redirect(url_for('users.face_auth', email=form.email.data, bio_desc=bio_desc))
        elif user.auth_type == 3:
            return redirect(url_for('users.otp', email=form.email.data))
        else:
            # bypass
            login_user(user)
            flash('Logged in successfully.', category='success')
            return redirect(url_for('core.index'))

    return render_template('login.html', form=form)

@users.route('/face_auth', methods=['GET', 'POST'])
def face_auth():
    if 'email' in request.args:
        email = request.args['email']
    else:
        return redirect(url_for('core.index'))

    form = FaceAuthForm()

    if form.validate_on_submit():
        user = User.query.filter_by(email=email).first()

        login_user(user)
        flash('Logged in successfully.', category='success')
        return redirect(url_for('core.index'))

    return render_template('face_auth.html', form=form)

@users.route("/logout")
@login_required
def logout():
    logout_user()
    session.pop("email",None)
    session.clear() #new if did not work delete it
    flash("Logout success", "success")
    return redirect(url_for('core.index'))


@users.route("/useraccount", methods=['GET'])
@login_required
def useraccount():
    form = GenerateWalletForm()

    current_user_email = str(current_user)
    user = User.query.filter_by(email=current_user_email).first()
    acc_add,pub_key,pri_key  =user.get_keys()
    balance = user.get_balance()

    form.email.data = current_user_email
    form.account_address.data = acc_add
    form.publickey.data = pub_key
    form.privatekey.data = '*'
    form.balance.data=balance

    return render_template('useraccount.html', form=form)


@users.route('/useraccount', methods=['POST'])
@login_required
def download_credentials():
    current_user_email = str(current_user)
    user = User.query.filter_by(email=current_user_email).first()

    if user.auth != 1:
        acc_add, pub_key,pri_key = user.get_keys()

        csv = 'Email Address,Account Address,Public Key,Private Key' + '\n' + current_user_email + ',' + acc_add + ',' + pub_key + ',' + pri_key

        user.privatekey = '*'
        user.auth = 1
        db.session.commit()

        return Response(
            csv,
            mimetype='text/csv',
            headers={'Content-disposition': 'attachment; filename=' + current_user_email + '_credentials.csv'})

    else:
        return 'Error: You have already downloaded your account credentials.'


@users.route('/otp', methods=['GET', 'POST'])
def otp():
    if 'email' in request.args:
        email = request.args['email']
    else:
        return redirect(url_for('core.index'))
    form = OTPForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=email).first()
        account_add = user.account_address
        user_account = codecs.decode(account_add[2:], 'hex')
        otp_token_hex = user_account.hex()
        otp_token = base64.b32encode(bytes.fromhex(otp_token_hex)).decode()
        totp = pyotp.TOTP(otp_token)
        if totp.verify(form.otp.data):
            if user.auth_type == 1:
                login_user(user)
                flash('Logged in successfully.', category='success')
                return redirect(url_for('core.index'))
            elif user.auth_type == 3:
                bio_desc = user.bio_desc
                return redirect(url_for('users.face_auth', email=email, bio_desc=bio_desc))
        else:
            flash('Invalid token.', category='error'),print("form.otp.data",form.otp.data)
            return render_template('otp.html', form=form)
    return render_template('otp.html', form=form)

@users.route("/secsetting", methods=['GET'])
@login_required
def secsetting():
    form = SecsettingForm()
    current_user_email = str(current_user)
    user = User.query.filter_by(email=current_user_email).first()
    account_add = user.account_address
    user_account = codecs.decode(account_add[2:], 'hex')
    otp_token_hex = user_account.hex()
    otp_token = base64.b32encode(bytes.fromhex(otp_token_hex)).decode()
    totp_url = pyotp.TOTP(otp_token).provisioning_uri(name=current_user_email, issuer_name="SecureWallet")
    auth_type = user.auth_type
    form.auth_type.data = auth_type
    form.otp_url.data = totp_url
    return render_template('secsetting.html' , form=form)
######
@users.route('/secsetting', methods=['POST'])
@login_required
def save_secsetting():
    current_user_email = str(current_user)
    user = User.query.filter_by(email=current_user_email).first()

    form = SecsettingForm()

    if form.validate_on_submit():
        bio_desc = form.bio_desc.data

        print('auth_type_radio', form.auth_type_radio.data)
        print('auth_type', form.auth_type.data)
        print('otp_url', form.otp_url.data)
        print('bio_desc', bio_desc)

        auth_type_result = form.auth_type_radio.data

        if auth_type_result == '0' or auth_type_result == '1' or auth_type_result == '2' or auth_type_result == '3':
            user.auth_type = int(auth_type_result)
            user.bio_desc = bio_desc
            db.session.commit()

            return 'Success: Your account security setting has been updated.'
        else:
            return 'Error: Wrong parameters.'


@users.route('/generate_transaction', methods=['GET', 'POST'])
@login_required
def generate_transaction():
    form = GenerateTransactionForm()
    if form.validate_on_submit():
        transaction = create_transaction(form)
        if transaction:
            if perform_transaction(transaction):
                flash('Transaction created successfully', category= 'success')
                return redirect(url_for('users.success'))
        else:
            flash('Error: Invalid Transaction!.', category= 'error')
            redirect(url_for('users.generate_transaction'))
    return render_template('generate_transaction.html', form=form)

@login_required
def create_transaction(form):

    sender_address = form.sender_address.data
    sender_private_key = form.sender_private_key.data.encode('ascii')
    balance = current_user.get_balance()
    recipient_address = form.recipient_address.data
    value = float(form.value.data)
    if sender_address != current_user.account_address:
        flash('Invalid Address:Please enter your correct address.',category= 'error')
    if User.get_balance_by_address(recipient_address) is None:
        flash("Recipient does not exist.", category='error')
    else:
        recipient_balance = User.get_balance_by_address(recipient_address)
        transaction = Transaction(sender_address, sender_private_key, recipient_address, value, balance,recipient_balance)
        return transaction

@login_required
def perform_transaction(transaction):
    if isinstance(transaction, Transaction) and transaction.transfer() and transaction.get_signature():
        user_email = current_user.email
        trans_id = uuid.uuid4().hex
        transaction_id = hashlib.sha256(trans_id.encode()).hexdigest()[:10]
        query = update(User).where(User.email == user_email).values(balance=transaction.balance)
        db.session.execute(query)
        user_trans=TransacForUsere(transaction_id,user_email, transaction.sender_address,transaction.recipient_address, transaction.value,transaction.get_signature())
        # add transaction
        db.session.add(user_trans)
        db.session.commit()
        # update recipient balance
        recipient = User.query.filter_by(account_address=transaction.recipient_address).first()
        recipient.balance = transaction.recipient_balance
        db.session.commit()

        return user_trans

@users.route('/success', methods=['GET'])
@login_required
def success():
    user_trans = current_user.get_last_transaction()

    return render_template('success.html',user_trans=user_trans)

@users.route('/get_user_transactions', methods=['GET'])
@login_required
def get_user_transactions():
    user_trans = current_user.get_all_transactions()
    return render_template('get_user_transactions.html',user_trans=user_trans)