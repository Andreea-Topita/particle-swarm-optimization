import numpy as np

class Particula:
    def __init__(self, dimensiune):
        self.pozitie = np.zeros(dimensiune)
        self.viteza = np.zeros(dimensiune)
        self.cost = float('inf')
        self.optim_personal_cost = float('inf')
        self.optim_personal_pozitie = np.zeros(dimensiune)



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

    def get_optim_social(self, index_particula):
        if self.modul == 'gbest':
            best_pos = self.roi[0].optim_personal_pozitie
            best_cost = self.roi[0].optim_personal_cost
            
            for p in self.roi:
                if p.optim_personal_cost < best_cost:
                    best_cost = p.optim_personal_cost
                    best_pos = np.copy(p.optim_personal_pozitie)
            return best_pos
            
        elif self.modul == 'lbest_social':
            indices = [index_particula - 1, index_particula, index_particula + 1]
            
            p_curenta = self.roi[index_particula]
            best_pos = np.copy(p_curenta.optim_personal_pozitie)
            best_cost = p_curenta.optim_personal_cost
            
            for idx in indices:
                real_idx = idx % self.nr_particule
                p_vecin = self.roi[real_idx]
                if p_vecin.optim_personal_cost < best_cost:
                    best_cost = p_vecin.optim_personal_cost
                    best_pos = np.copy(p_vecin.optim_personal_pozitie)
            return best_pos
                
        elif self.modul == 'lbest_geo':
            p_curenta = self.roi[index_particula]
            best_pos = np.copy(p_curenta.optim_personal_pozitie)
            best_cost = p_curenta.optim_personal_cost
            
            RAZA_VECINATATE = 6.0 
            
            for p_vecin in self.roi:
                dist = np.linalg.norm(p_curenta.pozitie - p_vecin.pozitie)
                
                if dist < RAZA_VECINATATE:
                    if p_vecin.optim_personal_cost < best_cost:
                        best_cost = p_vecin.optim_personal_cost
                        best_pos = np.copy(p_vecin.optim_personal_pozitie)
            return best_pos

        return np.zeros(self.dim)

    def optimizare(self):
        self.initializare()
        istoric_pozitii = []

        optim_social_pozitie = np.copy(self.roi[0].optim_personal_pozitie)
        optim_social_cost = self.roi[0].optim_personal_cost

        for p in self.roi:
            if p.optim_personal_cost < optim_social_cost:
                optim_social_cost = p.optim_personal_cost
                optim_social_pozitie = np.copy(p.optim_personal_pozitie)

        
        for t in range(self.nr_iteratii):
            frame_curent = []
            for p in self.roi:
                frame_curent.append(np.copy(p.pozitie))
            istoric_pozitii.append(frame_curent)

            self.w = 0.9 - ((0.9 - 0.4) * t / self.nr_iteratii)

            for i, p in enumerate(self.roi):
                optim_social = self.get_optim_social(i)
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                
                #v = w*v + c1*r1*(pbest-x) + c2*r2*(social-x)
                t_inertie = self.w * p.viteza
                t_cognitiv = self.c1 * r1 * (p.optim_personal_pozitie - p.pozitie)
                t_social = self.c2 * r2 * (optim_social - p.pozitie)
                
                p.viteza = t_inertie + t_cognitiv + t_social
                
                for d in range(self.dim):
                    p.viteza[d] = self.limiteaza(p.viteza[d], -self.v_max, self.v_max)
                
                p.pozitie = p.pozitie + p.viteza
                
                for d in range(self.dim):
                    p.pozitie[d] = self.limiteaza(p.pozitie[d], self.x_min, self.x_max)

                p.cost = self.func(p.pozitie)
                
                if p.cost < p.optim_personal_cost:
                    p.optim_personal_cost = p.cost
                    p.optim_personal_pozitie = np.copy(p.pozitie)
                    
                    if p.cost < optim_social_cost:
                        optim_social_cost = p.cost
                        optim_social_pozitie = np.copy(p.pozitie)

        return optim_social_pozitie, optim_social_cost, istoric_pozitii
    






