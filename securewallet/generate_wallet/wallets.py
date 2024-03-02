import binascii
import hashlib
from collections import OrderedDict
import ecdsa
from eth_utils import keccak
import secrets
import json
from ecdsa import SECP256k1
from eth_keys import keys

class InsufficientFundsException(Exception):
    """Exception raised when the value of the transaction is greater than the balance."""
    pass

class InvalidTransactionException(Exception):
    """Exception raised when the transaction is invalid (e.g. sender and recipient are the same or value is negative)."""
    pass



class Waccount:

    def __init__(self):
        priv = secrets.token_hex(32)
        private_key = "0x" + priv
        acct = keys.PrivateKey(bytes.fromhex(private_key[2:]))
        self.privatekey = private_key
        pub_key = acct.public_key
        self.publickey= pub_key.to_bytes().hex()
        address = keccak(pub_key.to_bytes()).hex()[-40:]
        self.account_address = "0x" + address
        self.balance = self.get_balance()

    def get_balance(self):
        balance = 100.0
        return balance

class Transaction:

    def __init__(self, sender_address, sender_private_key, recipient_address, value,balance,recipient_balance):
        self.sender_address = sender_address
        self.sender_private_key = sender_private_key
        self.recipient_address = recipient_address
        self.value = value
        self.balance= balance
        self.recipient_balance=recipient_balance


    def __repr__(self):
        return self.sender_address

    def __str__(self):
        return self.sender_address


    def signing_key(self):
        """Generate the signing key from the sender's private key."""
        #print (self.sender_private_key)
        priv=self.sender_private_key
        #print(type(priv))
        priv=int ((priv), base=16)
        priv=priv.to_bytes(32, 'big').hex()
        sk = ecdsa.keys.SigningKey.from_string(bytes.fromhex(priv), curve=ecdsa.SECP256k1,
                                               hashfunc=hashlib.sha256)
        return sk

    def verifing_key(self):
        """Generate the verifying key from the signing key."""
        sk= self.signing_key()
        vk = sk.get_verifying_key()
        return vk

    def transfer(self):
        """Transfer the value from the sender to the recipient.

        Returns:
            float: The updated balance after the transaction is completed.

        Raises:
            InsufficientFundsException: If the value of the transaction is greater than the balance.
            InvalidTransactionException: If the transaction is invalid (e.g. sender and recipient are the same or value is negative).
        """
        # Validate the value
        if self.value <= 0.0 :
            raise InvalidTransactionException("Invalid value.")

        # Check if the sender has sufficient funds

        if self.value > self.balance:
            raise InsufficientFundsException("Insufficient Funds.")

        # Check if the sender and recipient are the same
        elif self.sender_address == self.recipient_address:
            raise InvalidTransactionException("Sender and recipient are the same.")

        # Transfer the funds and return the updated balance
        self.balance -= self.value
        self.recipient_balance+= self.value
        return self.balance, self.recipient_balance, self.sender_address, self.value

    def trans_to_dict(self):
        """Convert the transaction data to a dictionary."""
        trans_data = OrderedDict([
            ("sender_publickey", self.sender_address),
            ("recipient_publickey", self.recipient_address),
            ("value", self.value)
        ])
        return trans_data

    def get_signature(self):
        """create the signature to sign the transaction with private key"""
        trans_data=str(self.trans_to_dict())

        encoded_transaction = json.dumps(trans_data,sort_keys=True).encode()
        sk = self.signing_key()
        signature = sk.sign(encoded_transaction,hashfunc=hashlib.sha256)
        return signature


    def verify_transaction_signature(self):
        """Verifies if the signature is correct. This is used to prove
        it's you (and not someone else) trying to do a transaction with your
        address. Called when a user tries to submit a new transaction.
        """
        trans_data = str(self.trans_to_dict())
        encoded_transaction =json.dumps(trans_data,sort_keys=True).encode()
        p= "0082f34da9ba08af487f3c996efd6e344aa7490bba7f3cdda78654550f1bbc735b621c0fd915814614a39e26544a479919365259f8958608d3d7a68b85ff843b"
        pubkey = (binascii.unhexlify(p.encode())).hex()
        signature = self.get_signature()
        vk = ecdsa.keys.VerifyingKey.from_string(bytes.fromhex(pubkey),curve= SECP256k1,hashfunc=hashlib.sha256)
        try:
            return vk.verify(signature, encoded_transaction,hashfunc=hashlib.sha256)
        except:
            return False

trans= Transaction(sender_address="0x3a7AeBE67FD9914aFfe643078bF1081273AfeE1e", sender_private_key="0xdc2294a5a45d92b887e9e41ca0382c04735f9454994feb5ea28c38177e719697", recipient_address="0x6e49074f3db06c496dda27ec9a6438ce7f8ebc4eabda6f235203f17c1467b686", value=10.0,balance=100.0,recipient_balance=100)

get_signature=trans.get_signature()

trans.verify_transaction_signature()












