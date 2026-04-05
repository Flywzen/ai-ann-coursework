def cost_function(route, distance_matrix):
    """
    Menghitung total jarak rute TSP (Cycle).
    Berdasarkan contoh rute A-B-C-D-A di slide [3].
    """
    total_distance = 0
    for i in range(len(route) - 1):
        # Tambahkan jarak dari kota i ke kota berikutnya
        total_distance += distance_matrix[route[i]][route[i+1]]
    
    # WAJIB: Tambahkan jarak kembali dari kota terakhir ke kota awal [4, 6]
    total_distance += distance_matrix[route[-1]][route]
    
    return total_distance
