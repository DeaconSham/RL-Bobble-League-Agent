import os

def get_positions():
    cur_path = os.path.realpath(__file__)
    cur_dir = os.path.dirname(cur_path)

    scene_f = open(cur_dir + "\\bobblefuck\\scene.tscn", "r")
    file_array = []
    line = scene_f.readline()

    while line != '':
        file_array.append(line)
        line = scene_f.readline()

    pos_ball = []

    for line in file_array:
        if line[:17] == '[node name=\"Ball\"':
            # some regex to come up with ball_extract
            # pos_ball = extracted_ball_pos
            pass
    
    if pos_ball == []:
        print('you goofed')
    
    pos_players = [[], [], []]