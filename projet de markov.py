from hmmlearn import hmm
import numpy as np

# Séquence d'observations obtenues à l'aide du site Investing
observations = np.array(['TH', 'TH', 'F', 'H', 'F', 'TF', 'F', 'H', 'TF', 'TF', 'TF', 'F', 'TF', 'M', 'M', 'TH', 'M', 'F', 'F', 'F', 'M', 'M'])

# Mapping des observations à des nombres
observations_map = {'TF': 0, 'F': 1, 'M': 2, 'H': 3, 'TH': 4}
observations_indices = np.array([observations_map[o] for o in observations]).reshape(-1, 1)

# Création d'un modèle HMM avec 3 états cachés
model = hmm.MultinomialHMM(n_components=3, n_iter=100)

# Initialisation aléatoire des paramètres du modèle
model.startprob_ = np.random.rand(3)
model.startprob_ /= model.startprob_.sum()

model.transmat_ = np.random.rand(3, 3)
model.transmat_ /= model.transmat_.sum(axis=1, keepdims=True)

model.emissionprob_ = np.random.rand(3, 5)
model.emissionprob_ /= model.emissionprob_.sum(axis=1, keepdims=True)

# Entraînement du modèle avec l'algorithme de Baum-Welch
model.fit(observations_indices)

# Prédiction des états pour les 10 prochains jours avec l'algorithme de Viterbi
predicted_states_indices = model.predict(observations_indices)
predicted_states = np.array(['Haut', 'Moyen', 'Bas'])[predicted_states_indices]

print("Paramètres du modèle estimés:")
print("Matrice de transition d'état :")
print(model.transmat_)
print("Matrice d'émission :")
print(model.emissionprob_)
print("Probabilités initiales des états :")
print(model.startprob_)
print("\nPrédiction des états pour les 10 prochains jours:")
print(predicted_states)

# Matrice obtenue à l'aide du site Investing, a chaque fois on compte me nombre de fois ou le marché passe d'un état a un autre et on normalise avec le nombre total d'état
A = [
    [0.25, 0.125, 0.625],
    [0.43, 0.14, 0.43],
    [0.25, 0.33, 0.42]
]

B= np.array([
    [0.1, 0.2, 0.3, 0.2, 0.2],
    [0.25, 0.1, 0.15, 0.25, 0.25],
    [0.4, 0.1, 0.05, 0.2, 0.25]
])
pi=[0.3,0.2,0.5]

# Algorithme de Baum-Welch pour estimer les paramètres du HMM
def baum_welch(observations, A, B, pi, n_iter=100):
    n_etats, n_observations = B.shape
    T = len(observations)

    for iteration in range(n_iter):
        # Etape E (expectation) - calcul des probabilités forward et backward
        alpha = np.zeros((T, n_etats))
        beta = np.zeros((T, n_etats))

        # Calcul des probabilités forward
        alpha[0] = pi * B[:, observations[0]]
        for t in range(1, T):
            alpha[t] = np.dot(alpha[t - 1], A) * B[:, observations[t]]

        # Calcul des probabilités backward
        beta[-1] = 1
        for t in range(T - 2, -1, -1):
            beta[t] = np.dot(A, (B[:, observations[t + 1]] * beta[t + 1]))

        # Calcul de la vraisemblance de l'observation
        likelihood = np.sum(alpha[-1])

        # Etape M (maximization) - mise à jour des paramètres
        xi = np.zeros((T - 1, n_etats, n_etats))
        for t in range(T - 1):
            xi[t] = (alpha[t][:, np.newaxis] * A * B[:, observations[t + 1]] * beta[t + 1]) / likelihood

        gamma = np.sum(xi, axis=1)
        A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:,
                                 np.newaxis]  # Utilisation de gamma[:-1] pour ajuster la longueur
        gamma = np.vstack(
            (gamma, np.zeros((1, n_etats))))  # Ajout d'une ligne de zéros à gamma pour ajuster la longueur
        gamma += 1e-6
        B = np.copy(B)
        for k in range(n_observations):
            B[:, k] = np.sum(gamma[observations == k], axis=0) / np.sum(gamma[:-1],
                                                                        axis=0)  # Utilisation de gamma[:-1] pour ajuster la longueur

        pi = alpha[0] / np.sum(alpha[0])

    return A, B, pi


# Algorithme de Viterbi pour prédire la séquence d'états cachés
def viterbi(observations, A, B, pi):
    n_etats, n_observations = B.shape
    T = len(observations)

    delta = np.zeros((T, n_etats))
    psi = np.zeros((T, n_etats), dtype=int)

    # Initialisation
    delta[0] = pi * B[:, observations[0]]


    for t in range(1, T):
        for j in range(n_etats):
            delta[t, j] = np.max(delta[t - 1] * A[:, j]) * B[j, observations[t]]
            psi[t, j] = np.argmax(delta[t - 1] * A[:, j])

    # Séquence d'états prédite
    etats_predits = np.zeros(T, dtype=int)
    etats_predits[-1] = np.argmax(delta[-1])

    for t in range(T - 2, -1, -1):
        etats_predits[t] = psi[t + 1, etats_predits[t + 1]]

    return etats_predits


# Séquence d'observations
observations = np.array([4, 4, 1, 3, 1, 0, 1, 3, 0, 0, 0, 1, 0, 2, 2, 4, 2, 1, 1, 1, 2,
                         2])  # Correspondant à [TH, TH, F, H, F, TF, F, H, TF, TF, TF, F, TF, M, M, TH, M, F, F, F, M, M] en numérique



# Estimation des paramètres du modèle avec l'algorithme de Baum-Welch
A, B, pi = baum_welch(observations, A, B, pi)

# Prédiction des états pour les 10 prochains jours avec l'algorithme de Viterbi
etats_predits = viterbi(observations, A, B, pi)

print("Matrice de transition d'état :")
for i in range(len(A)):
    print("État", i, ":", " ".join("{:.3f}".format(x) for x in A[i]))

print("\nMatrice d'émission :")
for i in range(len(B)):
    print("État", i, ":", " ".join("{:.3f}".format(x) for x in B[i]))

print("\nProbabilités initiales des états :")
print(" ".join("{:.3f}".format(x) for x in pi))

print("\nPrédiction des états pour les 10 prochains jours :")
print(" ".join("{:d}".format(x) for x in etats_predits))

