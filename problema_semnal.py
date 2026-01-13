import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from pso import PSO


DOMENIU = 10  
SURSA_REALA = np.array([8.0, 7.5]) 
PUTERE_EMISIE = 100.0
FACTOR_BRUIAJ=0.5
ZONE_BRUIAJ = [
    (4, 4, 2.0),
    (2, 8, 1.5)
]

mpl.rcParams['toolbar'] = 'None'

#distanta euclidiana intre 2 puncte
def calcul_distanta(p1, p2):
    return np.linalg.norm(p1 - p2)

def citeste_senzor(pozitie_robot):
    dist = calcul_distanta(pozitie_robot, SURSA_REALA)
    intensitate = PUTERE_EMISIE / (1 + dist**2)

    for (bx, by, raza_bruiaj) in ZONE_BRUIAJ:
        dist_bruiaj = calcul_distanta(pozitie_robot, np.array([bx, by]))
        if dist_bruiaj < raza_bruiaj:
            intensitate *= FACTOR_BRUIAJ
            
    return intensitate


def functie_fitness_pso(pozitie):
    return -citeste_senzor(pozitie)


if __name__ == "__main__":
    algoritm = PSO(
        functie_obiectiv=functie_fitness_pso,
        domeniu_min=0, domeniu_max=DOMENIU,
        dimensiune=2,
        nr_particule=30,
        nr_iteratii=60,
        w=0.7, c1=2, c2=2, modul='gbest'
    )
    best_pos, best_cost, istoric = algoritm.optimizare()
    print(f"Sursa gasita la: [{best_pos[0]:.2f}, {best_pos[1]:.2f}]")

    fig, ax = plt.subplots(figsize=(8, 8))
    
    X_grid = np.linspace(0, DOMENIU, 50)
    Y_grid = np.linspace(0, DOMENIU, 50)
    X, Y = np.meshgrid(X_grid, Y_grid)
    
    #distanta fiecarui punct din grid pana la sursa
    dist_sursa = np.sqrt((X - SURSA_REALA[0])**2 + (Y - SURSA_REALA[1])**2)
    Z = PUTERE_EMISIE / (1 + dist_sursa**2)

    for (bx, by, raza_bruiaj) in ZONE_BRUIAJ:
        dist_bruiaj = np.sqrt((X - bx)**2 + (Y - by)**2)
        mask = dist_bruiaj < raza_bruiaj
        Z[mask] *= FACTOR_BRUIAJ  

    heatmap = ax.contourf(X, Y, Z, levels=20, cmap='plasma', alpha=0.6)
    
    for (bx, by, raza_bruiaj) in ZONE_BRUIAJ:
        cerc = plt.Circle((bx, by), raza_bruiaj, color='gray', alpha=0.3, hatch='//')
        ax.add_patch(cerc)

    ax.plot(SURSA_REALA[0], SURSA_REALA[1], 'wX', markersize=15, markeredgecolor='black', label='Sursă Reală')

    roi_scatter = ax.scatter([], [], c='black', s=30, marker='o', label='Drone')
    lider_marker, = ax.plot([], [], 'y*', markersize=20, markeredgecolor='black', label='Lider')

    ax.set_xlim(0, DOMENIU)
    ax.set_ylim(0, DOMENIU)
    
    ax.set_title("Initializare...")
    ax.legend(loc='upper left')

    def update(frame):
        pozitii = istoric[frame]
    
        drone_x = [p[0] for p in pozitii]
        drone_y = [p[1] for p in pozitii]
        roi_scatter.set_offsets(np.c_[drone_x, drone_y])

        best_now_val = -float('inf')
        best_now_pos = [0, 0]
        
        for p in pozitii:
            val = citeste_senzor(p)
            if val > best_now_val:
                best_now_val = val
                best_now_pos = p

        lider_marker.set_data([best_now_pos[0]], [best_now_pos[1]])
        ax.set_title(f"Cautare Sursa Semnal\nIteratia {frame} | Semnal Maxim Detectat: {best_now_val:.1f}", fontsize=12)
        
        return roi_scatter, lider_marker

    ani = FuncAnimation(fig, update, frames=len(istoric), interval=400, blit=False)
    
    plt.show()
