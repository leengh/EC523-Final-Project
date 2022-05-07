import csv
from datetime import datetime

def save_csv(filename, episode, num_grasp, total_rewards):
    now = datetime.now()

    f = open(filename, 'r')
    csvreader = csv.reader(f)
    current_ros = [ro for ro in csvreader]

    f = open(filename, 'w')

    writer = csv.writer(f)
    writer.writerows(current_ros)

    current_time = now.strftime("%H:%M:%S")
    row =  [episode, num_grasp, total_rewards ,current_time]

    writer.writerow(row)
    f.close()
