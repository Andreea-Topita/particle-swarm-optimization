import numpy as np
import time
from pso import PSO
from problema_semnal import functie_fitness_pso, DOMENIU

FUNC_OBIECTIV = functie_fitness_pso
DIMENSIUNE = 2     
DOM_MIN = 0
DOM_MAX = DOMENIU
NR_RULARI = 5       


def ruleaza_test(tip_modul):
    semnale = []
    timpi = []

    for _ in range(NR_RULARI):
        start_time = time.time()
        
        alg = PSO(
            functie_obiectiv=FUNC_OBIECTIV,
            domeniu_min=DOM_MIN, 
            domeniu_max=DOM_MAX,
            dimensiune=DIMENSIUNE,
            nr_particule=30,   
            nr_iteratii=60,    
            w=0.7, c1=2, c2=2,
            modul=tip_modul
        )
        _, best_cost, _ = alg.optimizare()
        end_time = time.time()
        semnal_gasit = -best_cost 

        semnale.append(semnal_gasit)
        timpi.append(end_time - start_time)
        
    avg_score = np.mean(semnale)
    avg_time = np.mean(timpi)
    best_ever = np.max(semnale) 
    
    return avg_score, best_ever, avg_time

if __name__ == "__main__":
    print(f"\nRezultate Comparative: Problema2")
    print(f"Am avut {NR_RULARI} rulari independente.\n")
    
    s_avg, s_max, t_avg = ruleaza_test('gbest')
    print("GBEST (Global):")
    print(f"Semnal Mediu Obtinut: {s_avg:.4f}")
    print(f"Cel Mai Bun Semnal: {s_max:.4f}")
    print(f"Timp Mediu Executie: {t_avg:.4f} secunde\n")
    
    s_avg, s_max, t_avg = ruleaza_test('lbest_social')
    print("LBEST (Social):")
    print(f"Semnal Mediu Obtinut: {s_avg:.4f}")
    print(f"Cel Mai Bun Semnal: {s_max:.4f}")
    print(f"Timp Mediu Executie: {t_avg:.4f} secunde\n")
    
    s_avg, s_max, t_avg = ruleaza_test('lbest_geo')
    print("LBEST (Geografic)")
    print(f"Semnal Mediu Obtinut: {s_avg:.4f}")
    print(f"Cel Mai Bun Semnal: {s_max:.4f}")
    print(f"Timp Mediu Executie: {t_avg:.4f} secunde\n")