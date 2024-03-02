from flask import render_template,url_for,flash, redirect,request,Blueprint
from flask_login import current_user,login_required
from securewallet import db
from securewallet.models import WalletAccount
from securewallet.generate_wallet.forms import GenerateWalletForm
from securewallet.generate_wallet.wallets import *

from flask import render_template
wallet_account = Blueprint('wallet_account', __name__)

@wallet_account.route('/create', methods=['GET'])
@login_required
def create_wallet():


    form = GenerateWalletForm()

    if form.validate_on_submit():

        awallet_account = WalletAccount(account_address=form.account_address.data,
                                        privatekey=form.privatekey.data,
                                        user_id=current_user.id,
                                        publickey=current_user.publickey,
                                        balance= form.balance.data)
        db.session.add(awallet_account)
        db.session.commit()
        flash(" wallet account created")
        return redirect(url_for('core.index'))

    return render_template('create_wallet.html',form=form)





