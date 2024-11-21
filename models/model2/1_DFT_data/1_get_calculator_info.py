from ase.db import connect

# Connect to ase database
with connect('OH.db') as db:
	
	# Iterate through database rows
	for row in db.select():
	
		# Print calculator parameters of row
		print(row.calculator_parameters)
		
		# Stop loop
		break
