from flask import render_template, Blueprint, request,url_for, redirect
import joblib
import pandas as pd
from securewallet.core.forms import FeatureSelectionForm
from securewallet import db
from securewallet.models import BWAIRanking

core = Blueprint('core', __name__)

# Load the Random Forest model
model = joblib.load('random_forest_model.pkl')
DATA_FILE = 'test_dataset.csv'
def normalize_data():
    # Load data from CSV file
    df = pd.read_csv(DATA_FILE)

    # Convert 'No' to 0 and 'Yes' to 1 for specified columns
    binary_choice_columns = [
        'Support TOTP', 'Support Facial Recognition', 'Multiple Cryptocurrencies',
        'Non-Custodial', 'Custodial'
    ]
    for column in binary_choice_columns:
        if column in df.columns:
            df[column] = df[column].map({'No': 0, 'Yes': 1}).astype(int)
        else:
            print(f"Column {column} not found in the dataset.")

    # Normalize specified numerical columns
    numerical_columns = ['Wallet Age', 'Rating', 'Security Level']
    for column in numerical_columns:
        if column in df.columns:
            df[column] = (
                    (df[column] - df[column].min()) /
                    (df[column].max() - df[column].min())
            )
        else:
            print(f"Column {column} not found in the dataset.")

    return df


@core.route('/')
def index():
    return render_template('index.html')

@core.route('/ranking', methods=['GET', 'POST'])
def ranking():
    form = FeatureSelectionForm()
    if request.method == 'POST' and form.validate_on_submit():
        df = normalize_data()

        # Step 2: Generate predictions for all wallets
        predictions = model.predict(df)

        # Step 3: Update each wallet's rankings in the database
        for i, wallet_predictions in enumerate(predictions):

            wallet_identifier = df.iloc[i]['Wallet Identifier']
            wallet = BWAIRanking.query.filter_by(identifier=wallet_identifier).first()

            if wallet:
                wallet.update_rankings(
                    ranking=wallet_predictions[0],
                    totp_ranking=wallet_predictions[1],
                    facial_recognition_ranking=wallet_predictions[2],
                    cryptocurrencies_ranking=wallet_predictions[3],
                    age_ranking=wallet_predictions[4],
                    non_custodial_ranking=wallet_predictions[5],
                    custodial_ranking=wallet_predictions[6],
                    rating_ranking=wallet_predictions[7],
                    security_level_ranking=wallet_predictions[8]
                )
                db.session.add(wallet)

        db.session.commit()

        return redirect(url_for('index'))

    return render_template('ranking.html', form=form)


