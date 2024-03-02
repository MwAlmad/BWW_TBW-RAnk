from securewallet import*
from argparse import ArgumentParser
import ssl


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        db.session.commit()

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=8000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port
    cert_file = " /usr/local/etc/nginx/cert.pem"
    pkey_file = " /usr/local/etc/nginx/key.pem"

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    app.run(debug=True,host='127.0.0.1', port=port)
