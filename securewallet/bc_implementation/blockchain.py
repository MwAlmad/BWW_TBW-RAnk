from time import time
import time
from uuid import uuid4
from urllib.parse import urlparse
from datetime import datetime as dt
from securewallet.generate_wallet.wallets import*

class Block:
    def __init__(self, index, conf_transactions, timestamp, previous_hash, nonce=0):
        self.index = index
        self.transactions = conf_transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce


    def compute_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def __str__(self):
        return str("Block#: %s\nHash: %s\nPrevious: %s\nData: %s\nNonce: %s\n" % (
            self.index,
            self.compute_hash(),
            self.previous_hash,
            self.transactions,
            self.nonce
        )
                   )


class Blockchain:

    def __init__(self):
        transaction_id = str(uuid4()).replace('-', '')
        the_time = dt.now()
        timestamp = the_time.isoformat()
        self.chain=[]
        self.current_transactions= {}
        self.current_transactions['transaction_id'] = transaction_id
        self.current_transactions['timestamp'] = timestamp
        #self.current_transactions['trans_data'] = trans.trans_to_dict()
        self.gensis_block()
        # Generate random number to be used as node_id
        self.node_id = str(uuid4()).replace('-', '')
        self.nodes = set()

    def gensis_block(self):
        genesis_block = Block(0, [], time.time(), "0")
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)

    @property
    def last_block(self):
        return self.chain[-1]

    difficulty = 4

    def proof_of_work(self, block):
        """
                Simple Proof of Work Algorithm:
                 - Find a number p' such that hash(pp') contains leading 4 zeroes, where p is the previous p'
                 - p is the previous proof, and p' is the new proof
                :param last_proof: <int>
                :return: <int>
                """
        block.nonce = 456
        computed_hash = block.compute_hash()
        while not computed_hash.startswith('0'* Blockchain.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()
            return computed_hash


    def add_new_block(self, block,proof):
        """
                Create a new Block in the Blockchain
                :param proof: <int> The proof given by the Proof of Work algorithm
                :param previous_hash: (Optional) <str> Hash of previous Block
                :return: <dict> New Block
                """
        previous_hash = self.last_block.hash
        if previous_hash != block.previous_hash:
            return False
        if not self.is_valid_proof(block, proof):
            return False
        block.hash = proof
        self.chain.append(block)
        return True

    def is_valid_proof(self, block, block_hash):
        return (block_hash.startswith('0' * Blockchain.difficulty) and
                block_hash == block.compute_hash())

    def register_node(self, node_url): #04\01
        """
        Add a new node to the list of nodes
        """
        #Checking node_url has valid format
        parsed_url = urlparse(node_url)
        if parsed_url.netloc:
            self.nodes.add(parsed_url.netloc)
        elif parsed_url.path:
            # Accepts an URL without scheme like '192.168.0.5:5000'.
            self.nodes.add(parsed_url.path)
        else:
            raise ValueError('Invalid URL')



    # Manages transactions from wallet to another wallet
    def new_transaction(self, sender_publickey, trans):
        """
                Manages transactions from wallet to another wallet
                """
        transaction= trans.trans_to_dict()
        print(transaction)
        signature= trans.get_signatere()
        transaction_verification = trans.verify_transaction_signature(sender_publickey, signature, transaction)
        if transaction_verification:
            self.current_transactions.append({transaction})
            return self.last_block['index'] + 1,print(transaction)
        else:
            return False


    def mine(self):
        if not self.current_transactions:
            return False

        last_block = self.last_block

        new_block = Block(index=last_block.index + 1,
                          conf_transactions=self.current_transactions,
                          timestamp=time.time(),
                          previous_hash=last_block.hash)

        proof = self.proof_of_work(new_block)
        self.add_new_block(new_block, proof)
        self.current_transactions = []
        return new_block.index, print(new_block)


