import numpy as np

class Particula:
    def __init__(self, dimensiune):
        self.pozitie = np.zeros(dimensiune)
        self.viteza = np.zeros(dimensiune)
        self.cost = float('inf')
        self.optim_personal_cost = float('inf')


class PSO:
    def __init__(self, functie_obiectiv, domeniu_min, domeniu_max, dimensiune, 
                 nr_particule, nr_iteratii, w, c1, c2, modul='gbest'):
        
        self.func = functie_obiectiv
        self.x_min = domeniu_min    
        self.x_max = domeniu_max  
        self.dim = dimensiune      
        self.nr_particule = nr_particule
        self.nr_iteratii = nr_iteratii
        self.w = w                 
        self.c1 = c1             
        self.c2 = c2               
        self.modul = modul       
        
        #Vmax = alpha * (Xmax - Xmin) cu alpha aprox 0.2
        self.v_max = 0.2 * (domeniu_max - domeniu_min)
        self.roi = [] 

    def limiteaza(self, valoare, minim, maxim):
        if valoare < minim:
            return minim
        elif valoare > maxim:
            return maxim
        else:
            return valoare

    def initializare(self):
        self.roi = [] 

        for i in range(self.nr_particule):
            p = Particula(self.dim)
            for d in range(self.dim):
                p.pozitie[d] = np.random.uniform(self.x_min, self.x_max)
            
            p.cost = self.func(p.pozitie)
            p.viteza = np.zeros(self.dim)
            
            p.optim_personal_pozitie = np.copy(p.pozitie)
            p.optim_personal_cost = p.cost
            self.roi.append(p)

   