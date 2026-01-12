import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from pso import PSO

#fitness: lungimea traseului(suma de distante pe segmente) + penalizare mare daca vreun segment intersecteaza un obstacol 

START = np.array([0, 0])
STOP  = np.array([10, 10])
NR_PUNCTE_INTERMEDIARE = 3 

OBSTACOLE = [
    (4, 4, 1.5), 
    (7, 2, 1.0),
    (2, 8, 1.5),
    (8, 8, 1.0)
]

mpl.rcParams['toolbar'] = 'None'

#lungimea vectorului dintre 2 puncte
def distanta(p1, p2):
    return np.linalg.norm(p1 - p2)

def intersectie_segment_cerc(p1, p2, centru_cerc, raza):
    d = p2 - p1
    f = p1 - centru_cerc
    
    #produs scalar 
    a = np.dot(d, d)
    if a < 1e-6: 
        return False

    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - raza**2
    delta = b**2 - 4*a*c
    
    if delta < 0: 
        return False
    
    delta_sqrt = np.sqrt(delta)
    t1 = (-b - delta_sqrt) / (2*a)
    t2 = (-b + delta_sqrt) / (2*a)
    
    if (0 <= t1 <= 1) or (0 <= t2 <= 1): 
        return True
    
    return False

#functie de fitness(cost) pentru o particula
def cost_traseu(pozitie_particula):
    puncte = [START]

    for i in range(0, len(pozitie_particula), 2):
        puncte.append(np.array([pozitie_particula[i], pozitie_particula[i+1]]))
    puncte.append(STOP)
    #lista de puncte a traseului: start, puncte intermediare, stop

    cost = 0
    penalizare = 0
    for i in range(len(puncte) - 1):
        cost += distanta(puncte[i], puncte[i+1])
        for (ox, oy, oraza) in OBSTACOLE:
            if intersectie_segment_cerc(puncte[i], puncte[i+1], np.array([ox, oy]), oraza + 0.1):
                penalizare += 100000
                #sa nu treaca tangential prin obstacol, adaugam 0.1 la raza
                #daca intersecteaza : adaugam penalizare mare, solutie nevalida, pso trebuie sa evite
    return cost + penalizare
#pso cauta trasee scurte, dar pentru ca penalizarea e mare, prefera trasee mai lungi fara coliziuni

if __name__ == "__main__":
    algoritm = PSO(
        functie_obiectiv=cost_traseu,
        domeniu_min=0, domeniu_max=10,
        dimensiune=NR_PUNCTE_INTERMEDIARE * 2,
        nr_particule=50,   
        nr_iteratii=80, 
        w=0.7, c1=1.5, c2=1.5, modul='gbest'
    )
    
    best_pos, best_cost, istoric = algoritm.optimizare()
    print(f"Lungimea traseului final: {best_cost:.2f}")

    #animatie
    fig, ax = plt.subplots(figsize=(8, 8))
    
    #obstacolele
    for (ox, oy, oraza) in OBSTACOLE:
        ax.add_patch(plt.Circle((ox, oy), oraza, color='#ff4d4d', alpha=0.7))

    ax.add_patch(plt.Circle((START[0], START[1]), 0.4, color='#2ecc71', zorder=10))
    ax.text(START[0], START[1]-0.8, "START", ha='center', fontweight='bold', color='green')

    target_circle = plt.Circle((STOP[0], STOP[1]), 0.6, facecolor='none', edgecolor='blue', linewidth=3, zorder=10)
    ax.add_patch(target_circle)
    ax.text(STOP[0], STOP[1]+0.8, "TARGET", ha='center', fontweight='bold', color='blue')

    roi_scatter = ax.scatter([], [], c='#2c3e50', s=40, alpha=0.6, zorder=5, label='Roi')
    lider_line, = ax.plot([], [], color='blue', linewidth=4, alpha=0.8, zorder=4)

    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.set_xticks([]) 
    ax.set_yticks([])
    ax.grid(True, linestyle='-', alpha=0.2, color='black')

    info_text = ax.text(5, 10.5, "Initializare proces", ha='center', fontsize=14, fontweight='bold')

    #functie de update pentru animatie
    def update(frame):
        pozitii = istoric[frame]
        
        toate_x = []
        toate_y = []
        
        #best din iteratia curenta
        best_cost_now = float('inf')
        best_traseu_now = None

        for particula in pozitii:
            px = particula[0::2]
            py = particula[1::2] 
            toate_x.extend(px)
            toate_y.extend(py)

            c = cost_traseu(particula)
            if c < best_cost_now:
                best_cost_now = c
                best_traseu_now = ([START[0], *px, STOP[0]], [START[1], *py, STOP[1]])

        roi_scatter.set_offsets(np.c_[toate_x, toate_y])

        if best_traseu_now:
            lider_line.set_data(best_traseu_now[0], best_traseu_now[1])

        info_text.set_text(f"Iteratia: {frame} | Cost: {best_cost_now:.1f}")
        
        return roi_scatter, lider_line, info_text
    
    ani = FuncAnimation(fig, update, frames=len(istoric), interval=150, blit=False)
    
    plt.show()
