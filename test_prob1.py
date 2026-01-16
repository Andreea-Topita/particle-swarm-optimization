import unittest
import numpy as np
import problema_traseu as prob1
from problema_traseu import distanta, intersectie_segment_cerc, cost_traseu, NR_PUNCTE_INTERMEDIARE
from pso import PSO

class TestGeometrie(unittest.TestCase):
    def test_distanta_3_4_5(self):
        #distanta intre (0,0) si (3,4) trebuie sa fie 5, verificare functie distanta
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        #2 puncte 2D, distanta euclidiana
        self.assertAlmostEqual(distanta(p1, p2), 5.0, places=7)

    def test_intersectie_segment_cerc_true(self):
        #un segment care trece prin centru trebuie sa intersecteze cercul
        p1 = np.array([0.0, 0.0])
        p2 = np.array([10.0, 0.0])
        centru = np.array([5.0, 0.0])
        #segment orizontor, cerc centrat pe segment, intersecteaza
        raza = 1.0
        #verifica ca e adevarat
        self.assertTrue(intersectie_segment_cerc(p1, p2, centru, raza))

    def test_intersectie_segment_cerc_false(self):
        #un segment departe de cerc nu trebuie sa intersecteze
        p1 = np.array([0.0, 0.0])
        p2 = np.array([10.0, 0.0])
        centru = np.array([5.0, 5.0])
        #segment pe y=9, cerc centrat la (5,5), nu intersecteaza
        raza = 1.0
        #verifica ca e fals
        self.assertFalse(intersectie_segment_cerc(p1, p2, centru, raza))
        
    def test_intersectie_segment_degenerat(self):
        #daca segmentul are lungime 0 (p1 == p2), functia trebuie sa intoarca False
        p1 = np.array([1.0, 1.0])
        p2 = np.array([1.0, 1.0])
        centru = np.array([1.0, 1.0])
        raza = 2.0
        self.assertFalse(intersectie_segment_cerc(p1, p2, centru, raza))


class TestFitnessTraseu(unittest.TestCase):
    def test_penalizare_apare_la_coliziune(self):
        #waypoint-urile toate in (4,4), unde e un obstacol, deci coliziune sigura
        vec = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
        c = cost_traseu(vec)
        #particula are 6 valori: 3 puncte intermediare in (4,4), deci toate segmentele trec prin obstacol
        #obstacol in (4,4), deci coliziune, calcul fitnes, trebuie sa fie mai mare decat o penalizarea
        self.assertGreaterEqual(c, 100000)

    def test_cost_fara_obstacole_e_suma_distantelor(self):
        #scoatem obstacolele temporar ca sa verificam ca fitness ul devine doar suma distantelor
        old_obst = prob1.OBSTACOLE
        prob1.OBSTACOLE = []
        #old_obst salveaza obstacolele vechi
        try:
            vec = np.array([2.0, 0.0, 4.0, 0.0, 6.0, 0.0])
            #(0,0)->(2,0)->(4,0)->(6,0)->(10,10)
            expected = (
                distanta(np.array([0.0, 0.0]), np.array([2.0, 0.0])) +
                distanta(np.array([2.0, 0.0]), np.array([4.0, 0.0])) +
                distanta(np.array([4.0, 0.0]), np.array([6.0, 0.0])) +
                distanta(np.array([6.0, 0.0]), np.array([10.0, 10.0]))
            )
            #suma distantelor pe segmente, daca nu exista obstacole ar trebui sa fie fix suma distantelor
            self.assertAlmostEqual(cost_traseu(vec), expected, places=7)
        finally:
            prob1.OBSTACOLE = old_obst


class TestPSOIntegrat(unittest.TestCase):
    def setUp(self):
        self.dim = NR_PUNCTE_INTERMEDIARE * 2  
        #3 puncte -> 6 dimensiuni
        self.parametri_pso = dict(
            functie_obiectiv=cost_traseu,
            domeniu_min=0,
            domeniu_max=10,
            dimensiune=self.dim,
            nr_particule=15,
            nr_iteratii=20,
            w=0.7, c1=1.5, c2=1.5
        )

    def test_istoric_si_domeniu_corect(self):
        #verificare: best_pos dimensiune corecta,ramane in domeniu, istoric corect format
        np.random.seed(0)
        alg = PSO(modul='gbest', **self.parametri_pso)
        #fiecare cheie din dictionar si valoarea ei trimisa ca parametru
        best_pos, best_cost, istoric = alg.optimizare()

        #best_pos trebuie sa fie vector de dimensiune 6
        self.assertEqual(best_pos.shape, (self.dim,))
        #solutia finala raman in domeniu [0,10]
        self.assertTrue(np.all(best_pos >= 0) and np.all(best_pos <= 10))
        
        #istoricul trebuie sa aiba nr_iteratii intrari, fiecare cu nr_particule, fiecare cu dim valori
        self.assertEqual(len(istoric), self.parametri_pso["nr_iteratii"])
        self.assertEqual(len(istoric[0]), self.parametri_pso["nr_particule"])
        self.assertEqual(len(istoric[0][0]), self.dim)

        #best_cost trebuie sa fie fitness-ul lui best_pos, verificare consistenta: costul raportat e chia fitness pozitiei finale 
        self.assertAlmostEqual(best_cost, cost_traseu(best_pos), places=6)

    def test_ruleaza_toate_modurile(self):
        #verificam ca ruleaza fara erori in toate modurile
        for mode in ["gbest", "lbest_social", "lbest_geo"]:
            #sa fii sigur ca gbest, lbest_social, lbest_geo
            np.random.seed(42)
            alg = PSO(modul=mode, **self.parametri_pso)
            _, best_cost, _ = alg.optimizare()
            #rulare pso, doar best_cost ma intereseaza aici
            #best_Cost e un nr normal, nu NaN sau infinit
            self.assertTrue(np.isfinite(best_cost))

if __name__ == "__main__":
    unittest.main(verbosity=2)