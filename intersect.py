import numpy as np
import scipy as sc
import pylab as pl
from math import *
import math
from PIL import Image
import random
import time

'''
R : rayon sphere
C : Centre sphere
D : vecteur intersect
P : origine
'''

# calcul du delta et renvoie du resultat positif
def delta(a, b, c):
	d = b**2 - 4 * a * c
	if d < 0:
		return -1
	elif d == 0:
		result = -b / (2*a)
		return result
	else:
		result = (-b - math.sqrt(d)) / (2 * a)
		if result > 0:
			return result
		result = (-b + math.sqrt(d)) / (2 * a)
		if result > 0:
			return result
		return -1


# fonction de calcul d'intersection. Donne la distance entre l'origine et l'intersection la plus proche.
def intersect (R, C, P, D):
	A = D.dot(D)
	B = 2 * (np.dot(P,D) - np.dot(C,D))
	C = (C-P).dot(C-P) - R**2
	result = delta(A, B, C)
	if result == -1:
		return -1
	else:
		return result

# retourne le vecteur normalisé
def normalize(x):
	x = x / np.linalg.norm(x)
	return x

# classe sphere
class Sphere:
	def __init__(self, rayon, centre, meteriel):
		self.rayon = rayon
		self.centre = centre
		self.materiel = meteriel

# retourne en vecteur les cordonnées d'un point
def to_np_array(x, y, z):
	return np.array([x,y,z], dtype=float)

# retourne la distance entre 2 points
# a : arrivée ; b : départ
def distance_deux_point(a, b):
	return np.linalg.norm(a - b)

# parcourt la liste de sphere et calcule la distance la plus proche s'il y a une intersection
def sphere_proche(liste_sphere, p, d):
	distance_sphere_proche = 9999999
	change = 0
	for sphere in liste_sphere:
		distance = intersect(sphere.rayon, sphere.centre, p, d)
		distance = int(round(distance,0))
		if distance != -1 and distance < distance_sphere_proche:
			change = 1
			distance_sphere_proche = distance
			sphere_p = sphere
	if change == 1:
		return distance_sphere_proche, sphere_p
	else:
		return -1, None

# retourne la couleur correspondant à la bonne sphere
def couleur(liste_couleur, liste_sphere, sphere):
	return liste_couleur[liste_sphere.index(sphere_p)]

# pour avoir un format de couleur entre 0 et 1
def albedo(x):
	return np.array([x[0]/255,x[1]/255,x[2]/255], dtype=float)

# N vecteur normal de la surface de la sphere, W vecteur vers le capteur
def cos_theta(N, W):
	return N.dot(W)

# retourne la couleur en fonction de la puissance de la lumiere et atténuée par la distance
def diffusion(A, cos_theta, distance_lumiere_surface, puissance_lumiere):
	couleur = ((A*cos_theta)/np.pi) * 1/(distance_lumiere_surface**2) * puissance_lumiere
	return couleur

def reflexion (incoming, vect_norm_sphere):
	outcoming = (2 * ((-incoming)*vect_norm_sphere) * vect_norm_sphere + incoming)
	return outcoming




def couleur_mirroir(sphere_p, coord_point_sphere, rebond, material_surface, incoming, vect_norm_sphere, outcoming):
	if rebond >= 50 or material_surface == "diffus":
		color = couleur(liste_couleur, liste_sphere, sphere_p)
		return color
	rebond += 1

	# cherche une nouvelle intersection
	distance_sphere_proche, sphere_p = sphere_proche(liste_sphere, coord_point_sphere, outcoming)
	incoming = outcoming


	# a vérifier
	new_coord_point_sphere = coord_point_sphere + distance_sphere_proche
	new_norme_surface = normalize(new_coord_point_sphere - sphere_p.centre)
	new_outcoming = reflexion(incoming, new_norme_surface)

	couleur_mirroir(sphere_p, new_coord_point_sphere, rebond, sphere_p.materiel, incoming, new_norme_surface, new_outcoming)




'''
TEST LUMIERE ET DEGRADE
'''
img = Image.new( 'RGB', (1500, 1500), "black") # create a new black image
pixels = img.load() # create the pixel map
s1 = Sphere(3500, to_np_array(750,750,5000), "diffus")
s2 = Sphere(3500, to_np_array(-3500,750,2000), "diffus")
s3 = Sphere(3500, to_np_array(750,-3500,2000), "diffus")
s4 = Sphere(3500, to_np_array(5000,750,2000), "diffus")
s5 = Sphere(3500, to_np_array(750,5000,2000), "diffus")
s6 = Sphere(200, to_np_array(400,400,1500), "diffus")


liste_sphere = [s1, s2, s3, s4, s5, s6]
liste_couleur = [(255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0)]

point_perspective = np.array([750,750,-5000], dtype=float)
point_lumiere = np.array([750,750,1000], dtype=float)
puissance_lumiere = np.array([1000000000,1000000000,1000000000], dtype=float)

for i in range(img.size[0]):
    for j in range(img.size[1]):
    	origine = np.array([i,j,0], dtype=float)
    	d = origine - point_perspective
    	d = d / np.linalg.norm(d)
    	# sphere_p, la sphere qu'on a touché
    	distance_sphere_proche, sphere_p = sphere_proche(liste_sphere, origine, d)
    	# s'il y a eu une intersection
    	if distance_sphere_proche > 0:
    		# coordonnée x,y,z du point d'intersection (on rajoute *d a cause de la perspective)
    		coord_point_sphere = origine + distance_sphere_proche*d

    		# vecteur normal pointant vers la source de lumiere à partir de la sphere proche
	    	d_lumiere = point_lumiere - coord_point_sphere
	    	d_lumiere = normalize(d_lumiere)

	    	# vecteur normale à la surface de la sphere proche
	    	norme_surface = normalize(coord_point_sphere - sphere_p.centre)

	    	#pour décaler le point de la shpere légerement en dehors de la surface
	    	#evite pour un point de la sphere, en cherchant la source de lumiere, de rencontrer un obstacle qui mettrait un point d'ombre là ou il ne faut pas
	    	coord_point_sphere = coord_point_sphere + (norme_surface*4)
	    	# distance entre le point de la sphere et le point source de lumiere
	    	distance_sphere_lumiere = distance_deux_point(point_lumiere, coord_point_sphere)

	    	#cherche un obstacle entre le sphere et la lumiere
	    	intersect_lumiere, sphere_obstacle = sphere_proche(liste_sphere, coord_point_sphere, d_lumiere)
	    	# sert à calculer le cos_theta pour la diffusion
	    	norme_surface_capteur = normalize(point_perspective - coord_point_sphere)


	    	#s'il n'y a pas d'obstacle entre la sphere et la lumiere, ou que l'obstacle rencontré se trouve deriere la source de lumiere
	    	if intersect_lumiere == -1 or intersect_lumiere > distance_sphere_lumiere:
	    		rebond = 1
	    		outcoming = reflexion(d, norme_surface)
	    		# color = couleur_mirroir(sphere_p, coord_point_sphere, rebond, sphere_p.materiel, d, norme_surface, outcoming)
	    		couleur_sphere = couleur(liste_couleur, liste_sphere, sphere_p)
	    		# fonction de diffusion de la couleur
	    		e = diffusion(albedo(couleur_sphere), norme_surface.dot(norme_surface_capteur), distance_sphere_lumiere, puissance_lumiere)
	    		pixels[i,j] = (int(e[0]), int(e[1]), int(e[2]))
	    	else:
	    		# s'il y a eu un obstacle on récupere la couleur de la sphere
	    		couleur_sphere = couleur(liste_couleur, liste_sphere, sphere_p)
	    		# on met une couleur très aténuée
	    		pixels[i,j] = (int(couleur_sphere[0]/14), int(couleur_sphere[1]/14), int(couleur_sphere[2]/14))




img.show()
img.save("image.png", "PNG")