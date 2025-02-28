
class MacroSensitivities:
    def __init__(self):
        pass

    @staticmethod
    def calc_stress_level(row, seuil_taux=2.0, seuil_inflation=3.0, seuil_pib_très_faible=0.5):
        """
        Calcule le niveau de stress (1 à 5) en fonction des conditions suivantes :
        - Si (taux > seuil_taux et inflation > seuil_inflation) => stress 5.
        - Si (taux > seuil_taux ou inflation > seuil_inflation) => stress 4.
        - Si le PIB est très faible, on peut forcer un niveau de stress élevé.
        - Sinon, niveau de stress bas.
        Vous pouvez complexifier la logique en combinant plusieurs conditions.
        """
        stress = 1  # niveau par défaut "accommodant"

        # Condition sur taux et inflation
        if float(row['10Y Yield']) > seuil_taux and float(row['CPI']) > seuil_inflation:
            stress = 5
        elif float(row['10Y Yield']) > seuil_taux or float(row['CPI']) > seuil_inflation:
            stress = 4

        if float(row['GDP']) < seuil_pib_très_faible:
            # On peut, par exemple, augmenter le niveau de stress s'il n'est pas déjà à 5
            stress = max(stress, 5)

        # Vous pouvez ajouter d'autres conditions pour d'autres niveaux de stress
        return stress