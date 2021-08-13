import csv


current_data = [[25, 56], [1, -1], [27, 57], 95.036]
csv_file = open('employee_file.csv', mode='w')
for i in range(5):
    csv_file.write(f'{current_data[0]}, {current_data[1]}, {current_data[2]}, {current_data[3]} \n')

