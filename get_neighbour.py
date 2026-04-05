import random

def get_neighbor(route):
    """
    Menghasilkan kandidat solusi baru dengan menukar dua kota secara acak.
    Sesuai mekanisme 'Swap' pada simulasi SA slide [3].
    """
    new_route = list(route) # Salin rute agar tidak mengubah rute asli
    
    # Pilih dua indeks secara acak (bukan kota awal jika kota awal dikunci)
    idx1, idx2 = random.sample(range(len(route)), 2)
    
    # Lakukan pertukaran (Swap) [3]
    new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
    
    return new_route

