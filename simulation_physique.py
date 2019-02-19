import numpy as np
from matplotlib import pyplot as plt

# x : horizontal
# y : horizontal
# z : vertical

# vitesse terminale en m/s
vitesse_terminale_m1 = 50
vitesse_terminale_m2 = 80
# en Kg
m1 = 90
m2 = 90
# vitesse vent m/s
vitesse_vent = np.array([50,0,0], dtype=float)
# intervalle de saut entre les jumpers en seconde
intervalle_saut = 7

# variation du temps en seconde
delta_t = 0.01
# vecteur vers le 'bas'
z = np.array([0,0,1], dtype=float)
# constante gravitationelle
g = 9.81
# constante de frottement
Cd_m1 = (m1 * g)/(vitesse_terminale_m1**2)
Cd_m2 = (m2 * g)/(vitesse_terminale_m2**2)
# temps initialisé à 0
t = 0

# vitesse initiale 41 m/s
vitesse_initiale = np.array([41,0,0], dtype=float)
# position initiale en mètre
pos_initiale_m1 = np.array([0,0,4000], dtype=float)
pos_initiale_m2 = np.array([vitesse_initiale[0] * intervalle_saut,0,4000], dtype=float)


position_m1 = pos_initiale_m1
position_m2 = pos_initiale_m2
vitesse_m1 = vitesse_initiale
vitesse_m2 = vitesse_initiale


# #pour visualiser z (altitude en fonction du temps)
# graph_m1 = np.array([0,4000])
# graph_m2 = np.array([intervalle_saut,4000])
#pour visualiser x en fonction du temps
graph_m1 = np.array([0,0])
graph_m2 = np.array([0,vitesse_initiale[0] * intervalle_saut])


def frottement(Cd ,v):
	frottement = -Cd * np.linalg.norm(v) * v
	return frottement

def delta_acceleration(m, g, z, v, Cd):
	a = ((-m * g * z) + frottement(Cd, v)) / m
	return a

def delta_vitesse(delta_t, v, a):
	vitesse = delta_t * a + v
	return vitesse

# en supposant que l'avion se dirige contre le vent d'où le : (- (vitesse_vent) * delta_t)
def delta_position(delta_t, p, v, vitesse_vent):
	position = delta_t * v + p - (vitesse_vent) * delta_t
	return position

while position_m1[2] > 0:
	if position_m1[2] > 0:
		acceleration_m1 = delta_acceleration(m1, g, z, vitesse_m1, Cd_m1)
		vitesse_m1 = delta_vitesse(delta_t, vitesse_m1, acceleration_m1)
		position_m1 = delta_position(delta_t, position_m1, vitesse_m1, vitesse_vent)

	if position_m2[2] > 0:
		acceleration_m2 = delta_acceleration(m2, g, z, vitesse_m2, Cd_m2)
		vitesse_m2 = delta_vitesse(delta_t, vitesse_m2, acceleration_m2)
		position_m2 = delta_position(delta_t, position_m2, vitesse_m2, vitesse_vent)

	t += delta_t

	tmp_m1 = np.array([t,position_m1[0]])
	graph_m1 = np.vstack((graph_m1, tmp_m1))

	tmp_m2 = np.array([t + intervalle_saut,position_m2[0]])
	graph_m2 = np.vstack((graph_m2, tmp_m2))

x, y = graph_m1.T
t, w = graph_m2.T
plt.title("Variation de la position en x avec vent contraire")
plt.ylabel("position en x")
plt.xlabel("temps en seconde")
plt.plot(x,y,'-', label="vitesse terminale 50")
plt.plot(t,w,'-', label="vitesse terminale 80")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.show()