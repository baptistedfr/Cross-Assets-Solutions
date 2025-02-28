import yfinance as yf
import pandas as pd
import time

# Dictionnaire de correspondance Exchange -> suffixe Yahoo
exchange_suffix_map = {
    "LN": ".L",  # London
    "IM": ".MI",  # Borsa Italiana (Milan)
    "GY": ".DE",  # Xetra (Allemagne)
    "NA": ".AS",  # Euronext Amsterdam
    "FP": ".PA",  # Euronext Paris
    "BB": ".BR",  # Euronext Bruxelles
    "SM": ".MC",  # BME (Espagne)
    "DC": ".CO",  # Copenhague (Danemark)
    "AV": ".VI",  # Vienne (Autriche)
    "PW": ".WA",  # Varsovie (Pologne)
    "GA": ".AT",  # Athènes (Grèce)
    "ID": ".IR",  # Euronext Dublin (Irlande)
    "SS": ".ST",  # Stockholm (Suède) — attention : “SS” peut aussi désigner Shanghai dans d’autres contextes
    "SE": ".SW",  # SIX Swiss Exchange (Suisse) — parfois “.SW” ou “.VX”
    "NO": ".OL"  # Oslo (Norvège)
    # Ajoutez d’autres correspondances si besoin (ex. "SW" -> ".SW", etc.)
}


def build_yahoo_ticker(symbol, exchange):
    """
    Construit le ticker Yahoo à partir du Symbol et de l'Exchange.
    Si l'Exchange n'est pas connu, on renvoie symbol sans suffixe (risque d'être introuvable).
    """
    suffix = exchange_suffix_map.get(exchange, "")
    return symbol + suffix

# Exemple : vous pouvez mettre votre longue liste dans un fichier texte,
# ou directement dans un tableau Python. Ici on suppose que c'est un fichier.
input_file = "tickers.txt"  # remplacez par le nom de votre fichier
output_file = "converted_prices.csv"

# Lecture des tickers depuis le fichier
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.read().strip().split("\n")

tickers_raw = lines[0].split('\t')

# Conversion de chaque ligne
yahoo_tickers = []
for item in tickers_raw:
    # item ressemble à "0876218D LN Equity"
    parts = item.split()
    if len(parts) >= 2:
        symbol, exchange = parts[0], parts[1]
        yahoo_ticker = build_yahoo_ticker(symbol, exchange)
        yahoo_tickers.append(yahoo_ticker)

# Affiche la liste convertie pour contrôle

print("Exemples de tickers convertis pour Yahoo :")
for t in yahoo_tickers[:10]:
    print(t)
print(f"Nombre total de tickers convertis : {len(yahoo_tickers)}")

CHUNK_SIZE = 20  # Nombre de tickers par batch
BASE_SLEEP = 15  # Pause de base (secondes) après un batch réussi
MAX_RETRIES = 5  # Nombre max de tentatives si rate-limited

dataframes = []

for i in range(0, len(yahoo_tickers), CHUNK_SIZE):
    batch_tickers = yahoo_tickers[i: i + CHUNK_SIZE]

    attempts = 0
    success = False
    while attempts < MAX_RETRIES and not success:
        try:
            # Récupération sur 5 ans
            data_chunk = yf.download(batch_tickers, period="5y")["Close"]
            dataframes.append(data_chunk)
            success = True  # Téléchargement réussi, on peut sortir de la boucle
        except yf.shared.exceptions.YFRateLimitError:
            attempts += 1
            wait_time = BASE_SLEEP * attempts
            print(f"Rate limited (tentative {attempts}/{MAX_RETRIES}). Pause de {wait_time} s...")
            time.sleep(wait_time)
        except Exception as e:
            # Si l'erreur est d'un autre type, on l'affiche et on arrête
            print(f"Erreur inattendue : {e}")
            break

    # Si on a réussi, on attend BASE_SLEEP secondes avant de passer au batch suivant
    if success:
        time.sleep(BASE_SLEEP)
    else:
        print(f"Impossible de récupérer le batch {batch_tickers} après {MAX_RETRIES} tentatives.")
        # On peut décider de continuer ou d'arrêter. Ici, on continue simplement.

# Concaténer tous les DataFrames
if dataframes:
    final_data = pd.concat(dataframes, axis=1)
    # Éliminer d'éventuelles colonnes dupliquées
    final_data = final_data.loc[:, ~final_data.columns.duplicated()]

    # Sauvegarde
    final_data.to_csv("converted_prices.csv")
    print("Extraction terminée. Données enregistrées dans 'converted_prices.csv'.")
else:
    print("Aucune donnée récupérée.")