pos_data = []
# Some typical values
d = 2  # Could also be 3
vol_ext = (1000, 500)  # If d = 3, this will have another entry
ratio = [5.0, 8.0]  # Again, if d = 3, it will have another entry

for i in range(d):
    pos_data.append(np.zeros(vol_ext))
vol_ext = (1000, 500)