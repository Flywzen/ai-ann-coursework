import numpy as np
import math
import random

def simulated_annealing():
    # 1. INITIALIZATION [9, 10]
    # S = Solusi awal (misal: urutan kota acak untuk TSP)
    current_state = initialize_solution() 
    current_cost = cost_function(current_state)
    
    # Set hyperparameter [10, 11]
    T = 100.0          # Initial Temperature (T0)
    T_min = 0.001      # Stopping criteria (Suhu minimal)
    alpha = 0.9        # Cooling Rate (0.8 - 0.99) [1]
    max_iter = 1000    # Batas maksimal iterasi
    
    best_state = current_state
    best_cost = current_cost

    # 2. MAIN LOOP [1, 9]
    while T > T_min:
        for i in range(max_iter):
            # 2.1. GENERATE NEIGHBOR [9, 10]
            # Modifikasi solusi (misal: swap dua elemen)
            neighbor_state = get_neighbor(current_state)
            neighbor_cost = cost_function(neighbor_state)
            
            # 2.2. EVALUATE COST DIFFERENCE (Delta E) [8, 9]
            # Dalam minimasi: Delta E = Cost_Baru - Cost_Lama
            delta_e = neighbor_cost - current_cost
            
            # 2.3. ACCEPTANCE PROBABILITY [8, 9]
            if delta_e < 0:
                # Terima jika solusi lebih baik
                current_state = neighbor_state
                current_cost = neighbor_cost
                
                # Update global best
                if current_cost < best_cost:
                    best_state = current_state
                    best_cost = current_cost
            else:
                # Jika solusi lebih buruk, hitung probabilitas [8, 12]
                # P = exp(-Delta E / T)
                acceptance_p = math.exp(-delta_e / T)
                r = random.random()
                
                if r < acceptance_p:
                    # Terima solusi buruk untuk EKSPLORASI [5, 12]
                    current_state = neighbor_state
                    current_cost = neighbor_cost
        
        # 2.4. COOLING SCHEDULE [1, 9]
        # T = alpha * T (Penurunan suhu secara geometris)
        T = T * alpha
        
    # 3. RETURN RESULT [9]
    return best_state, best_cost

# Catatan: Fungsi cost_function() dan get_neighbor() 
# harus disesuaikan dengan kasus spesifik (TSP, Knapsack, dll) [2, 13].


--------------------------------------------------------------------------------
