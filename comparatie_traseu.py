import numpy as np
import time
from pso import PSO
from problema_traseu import cost_traseu, NR_PUNCTE_INTERMEDIARE

FUNC_OBIECTIV = cost_traseu
DIMENSIUNE = NR_PUNCTE_INTERMEDIARE * 2
DOM_MIN = 0
DOM_MAX = 10
NR_RULARI = 5

def ruleaza_test(tip_modul):
    costuri = []
    timpi = []
    
    for _ in range(NR_RULARI):
        start_time = time.time()
        
        alg = PSO(
            functie_obiectiv=FUNC_OBIECTIV,
            domeniu_min=DOM_MIN, 
            domeniu_max=DOM_MAX,
            dimensiune=DIMENSIUNE,
            nr_particule=40,   
            nr_iteratii=100,   
            w=0.7, c1=1.5, c2=1.5,
            modul=tip_modul
        )
        
        _, best_cost, _ = alg.optimizare()
        
        end_time = time.time()
        
        costuri.append(best_cost)
        timpi.append(end_time - start_time)
    
    #costul mediu, cel mai bun cost si timpul mediu
    avg_cost = np.mean(costuri)
    avg_time = np.mean(timpi)
    best_ever = np.min(costuri)
    
    return avg_cost, best_ever, avg_time

if __name__ == "__main__":
    print(f"\nRezultate Comparative: Problema 1 (Traseu)")
    print(f"Am avut {NR_RULARI} rulari independente.\n")
    
    c_avg, c_min, t_avg = ruleaza_test('gbest')
    print("GBEST (Global):")
    print(f"Cost Mediu Obtinut: {c_avg:.4f}")
    print(f"Cel Mai Bun Cost:   {c_min:.4f}")
    print(f"Timp Mediu Executie:  {t_avg:.4f} secunde\n")
    
    c_avg, c_min, t_avg = ruleaza_test('lbest_social')
    print("LBEST (Social):")
    print(f"Cost Mediu Obtinut: {c_avg:.4f}")
    print(f"Cel Mai Bun Cost:   {c_min:.4f}")
    print(f"Timp Mediu Executie:  {t_avg:.4f} secunde\n")
    
    c_avg, c_min, t_avg = ruleaza_test('lbest_geo')
    print("LBEST (Geografic):")
    print(f"Cost Mediu Obtinut: {c_avg:.4f}")
    print(f"Cel Mai Bun Cost:   {c_min:.4f}")
    print(f"Timp Mediu Executie:  {t_avg:.4f} secunde\n")
    
    print("LBEST converge de obicei mai lent, dar evita minimele locale (obstacolele) mai bine.")
