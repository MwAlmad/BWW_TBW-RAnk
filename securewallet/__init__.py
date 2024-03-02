import os
from flask_migrate import Migrate
from flask_login import LoginManager
from flask import Flask, session
from flask_sqlalchemy import SQLAlchemy
from datetime import timedelta
import secrets
from flask_qrcode import QRcode
from securewallet.bc_implementation.blockchain import Blockchain
from flask_cors import CORS

app = Flask(__name__, static_folder='static')

CORS(app)

app.config['SECRET_KEY'] = secrets.token_hex(32)


basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
Migrate(app, db)

QRcode(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "users.login"

from securewallet.core.views import core
from securewallet.wusers.views import users
from securewallet.generate_wallet.views import wallet_account
from securewallet.error_pages.handlers import error_pages
from securewallet.bc_implementation.views import bc_implementation

app.register_blueprint(users)
app.register_blueprint(core)
app.register_blueprint(wallet_account)
app.register_blueprint(error_pages)
app.register_blueprint(bc_implementation)

@app.before_request
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=5)
    session.modified = True
