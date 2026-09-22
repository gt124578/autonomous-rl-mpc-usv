import rosbag
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= CONFIGURATION DU TEST =================

# ================= CONFIGURATION DU TEST 3 =================
BAG_FILE = 'test_2.bag'  # Nom de ton fichier bag pour ce test
TITLE = 'Test 2 - Corridor'

# Cible (Waypoint) et Départ
TARGET_X = 100.0
TARGET_Y = 40.0

START_X = 100.0
START_Y = 0.0

# Génération du mur transversal de (75, 84) à (75, 114)
# On crée une ligne de petits obstacles serrés (tous les 1 mètre par exemple)
#OBSTACLES = [(-88.0, 68.0), (-106.0, 68.0), (-70.0, 68.0), (-61.0, 55.0), (-67.0, 44.0), (-75.0, 55.0), (-67.0, 44.0), (-86.0, 42.0), (-78.0, 30.0), (-80.0, 30.0), (-82.0, 30.0), (-84.0, 30.0), (-86.0, 30.0), (-105.0, 35.0), (-102.0, 53.0), (-115.0, 55.0)]
OBSTACLES = [(94.0, 10.0), (94.0, 15.0),(94.0, 20.0),(94.0, 25.0),(94.0, 30.0),(106.0, 10.0),(106.0, 15.0),(106.0, 20.0),(106.0, 25.0),(106.0, 30.0)]

"""
wall_x = 75.0
y_start = 84.0
y_end = 114.0
step = 1.0  # Espacement entre chaque point constituant le mur

current_y = y_start
while current_y <= y_end:
    OBSTACLES.append((wall_x, current_y))
    current_y += step"""
# =========================================================

"""
# Génération test 4
# ================= CONFIGURATION DU TEST 4 =================
BAG_FILE = 'test_4.bag'  # Nom de ton fichier bag pour ce test
TITLE = 'Test 4 - Major Central Obstacle'

# Cible (Waypoint)
TARGET_X = -60.0
TARGET_Y = 100.0

# Position de départ (pour le point vert sur la carte)
START_X = 0.0
START_Y = 100.0
# Chaque obstacle est espacé par exemple de 2 mètres (à adapter selon ta simulation)
OBSTACLES = []
grid_center_x = -30.0
grid_center_y = 100.0
grid_size = 7
spacing = 2.0  # Distance entre chaque plot/obstacle de la grille

offset = (grid_size - 1) * spacing / 2.0
for i in range(grid_size):
    for j in range(grid_size):
        obs_x = grid_center_x - offset + (i * spacing)
        obs_y = grid_center_y - offset + (j * spacing)
        OBSTACLES.append((obs_x, obs_y))"""
# =========================================================
#OBSTACLES = [(-30.0, 100.0), (30.0, -15.0), (50.0, -15.0), (45.0, -5.0), (40.0, 4.0), (60.0, 4.0)]
# =========================================================

def plot_bag(bag_file, title_prefix):
    print(f"Analyse du fichier {bag_file}...")
    
    boat_x, boat_y, time_states = [], [], []
    cmd_left, time_left = [], []
    cmd_right, time_right = [], []

    # Ouverture du rosbag
    bag = rosbag.Bag(bag_file)
    t0 = None

    for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states', '/myboat/thrusters/left_thrust_cmd', '/myboat/thrusters/right_thrust_cmd']):
        if t0 is None:
            t0 = t.to_sec()
        current_time = t.to_sec() - t0

        if topic == '/gazebo/model_states':
            try:
                # On récupère le X, Y du bateau
                idx = msg.name.index("myboat")
                boat_x.append(msg.pose[idx].position.x)
                boat_y.append(msg.pose[idx].position.y)
                time_states.append(current_time)
            except ValueError:
                pass
        elif topic == '/myboat/thrusters/left_thrust_cmd':
            cmd_left.append(msg.data)
            time_left.append(current_time)
        elif topic == '/myboat/thrusters/right_thrust_cmd':
            cmd_right.append(msg.data)
            time_right.append(current_time)
            
    bag.close()
    print("Données extraites ! Génération des graphiques...")

    # --- GRAPHIQUE 1 : TOP-DOWN MAP ---
    plt.figure(figsize=(10, 6))
    
    # Dessin de la trajectoire et points clés
    plt.plot(boat_x, boat_y, 'b-', label='Trajectoire USV', linewidth=2)
    if len(boat_x) > 0:
        plt.plot(boat_x[0], boat_y[0], 'go', label='Départ', markersize=8)
    plt.plot(TARGET_X, TARGET_Y, 'y*', label='Cible', markersize=15)

    # Dessin des obstacles et de la zone de sécurité du Shield (2m)
    for obs in OBSTACLES:
        circle = plt.Circle(obs, 1.0, color='r', fill=True, alpha=0.7, label='Obstacle')
        safe_zone = plt.Circle(obs, 3.0, color='r', fill=False, linestyle='--', alpha=0.5, label='Marge Sécurité (2m)')
        plt.gca().add_patch(circle)
        plt.gca().add_patch(safe_zone)

    # Éviter les doublons dans la légende
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.xlabel('Position X (m)')
    plt.ylabel('Position Y (m)')
    plt.title(f'{title_prefix} : Top-Down Map')
    plt.grid(True)
    plt.axis('equal')
    
    map_filename = f'{os.path.splitext(bag_file)[0]}_map.pdf'
    plt.savefig(map_filename, bbox_inches='tight')
    print(f"Carte sauvegardée sous {map_filename}")

    # --- GRAPHIQUE 2 : COMMANDES MOTEURS (CHATTERING) ---
    plt.figure(figsize=(10, 4))
    plt.plot(time_left, cmd_left, 'r-', label='Moteur Gauche', alpha=0.8)
    plt.plot(time_right, cmd_right, 'b-', label='Moteur Droit', alpha=0.8)
    plt.xlabel('Temps (s)')
    plt.ylabel('Commande PWM (-1 à 1)')
    plt.title(f'{title_prefix} : Profil des Commandes (RL + Shield)')
    plt.legend()
    plt.grid(True)
    
    cmd_filename = f'{os.path.splitext(bag_file)[0]}_cmds.pdf'
    plt.savefig(cmd_filename, bbox_inches='tight')
    print(f"Profil sauvegardé sous {cmd_filename}")

if __name__ == '__main__':
    plot_bag(BAG_FILE, TITLE)