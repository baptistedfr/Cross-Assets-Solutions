
class MacroSensitivities:
    def __init__(self):
        pass

    @staticmethod
    def calc_stress_level(row, seuil_pib_très_faible=0.5):
        """
        Calcule le niveau de stress (1 à 5) en fonction des conditions suivantes :
        - Si (taux > seuil_taux et inflation > seuil_inflation) => stress 5.
        - Si (taux > seuil_taux ou inflation > seuil_inflation) => stress 4.
        - Si le PIB est très faible, on peut forcer un niveau de stress élevé.
        - Sinon, niveau de stress bas.
        Vous pouvez complexifier la logique en combinant plusieurs conditions.
        """
        stress = 1  # niveau par défaut "très accommodant"

        if row['Refi Rate'] > row['Refi Rate_ema']:
            stress = 4

        if float(row['CPI']) > float(row['CPI_ema']):
            stress = 2
            if float(row['10Y Yield']) > float(row['10Y Yield_ema']):
                stress = 4
        else:
            stress = 2


        if float(row['GDP']) < seuil_pib_très_faible:
            stress = max(stress, 2)


        # Vous pouvez ajouter d'autres conditions pour d'autres niveaux de stress
        return stress