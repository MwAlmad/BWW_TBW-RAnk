import json

from flask_login import current_user,login_required
from securewallet import db
from securewallet.models import WalletAccount
from securewallet.generate_wallet.forms import GenerateWalletForm
from securewallet.generate_wallet.wallets import *
from securewallet.bc_implementation.forms import AddNodeForm,BlockForm
from securewallet.models import AddNode,Block
from securewallet.bc_implementation.blockchain import Blockchain,Block

from flask import render_template
from flask import render_template, url_for, flash, redirect, request, Blueprint, session, Response

blockchain=Blockchain()
#block=Block()


bc_implementation = Blueprint('bc_implementation', __name__)


@bc_implementation.route('/blockchain/main')
def main_blockchain():
    return render_template('./main_blockchain.html')



@bc_implementation.route('/configure')
def get_chain():
    chain_data = []
    for block in blockchain.chain:
        chain_data.append(block.__dict__)
    return json.dumps({"length": len(chain_data),
                       "chain": chain_data})


@bc_implementation.route('/transactions/get', methods=['GET'])
def get_transactions():
    # Get transactions from transactions pool
    transactions = blockchain.current_transactions
    return transactions

@bc_implementation.route(('/transactions/mine'), methods=['GET'])
def mine_transactions():
    form = BlockForm()

    last_block = blockchain.chain[-1]
    #nonce = blockchain.proof_of_work()


    mined_block = BlockForm(index=form.block_index,
                            transaction=form.transactions.data,
                            timestamp=form.date.data,
                            previous_hash=form.previous_hash.data)

    flash('New Block Forged', category='success')

    db.session.commit()

    return render_template('mine_transactions.html',mined_block=mined_block,form=form)


@bc_implementation.route('/node/add', methods=['GET','POST', ])
def add_nodes():
    """add a node."""

    form = AddNodeForm()

    if form.validate_on_submit():
        nodes = AddNode.query.filter_by(nodes_url=form.nodes_url.data).first()

        if nodes is not None:
            flash('This node already exists, Please enter a different url.', category='error')
            return redirect(url_for('bc_implementation.add_nodes'))

        nodes = AddNode(nodes_url=form.nodes_url.data)
        db.session.add(nodes)
        db.session.commit()
        flash('Thanks for registering! Now you can mine transactions!', category='success')
        return redirect(url_for('bc_implementation.add_nodes'))

    return render_template('add_node.html', form=form)




