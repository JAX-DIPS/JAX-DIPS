"""
PQR format: 
Field_name Atom_number Atom_name Residue_name Chain_ID Residue_number X Y Z Charge Radius
"""

import pdb


class load(object):
	def __init__(self, address, file_name, verbose=True):
		self.address = address
		self.file_name = file_name
		self.verbose = verbose
		self.atoms = {'x':[], 'y':[], 'z':[], 'q':[], 'R':[]}
		self.read()

	def read(self):
		fil = open(self.address + "/" + self.file_name, "r")
		ln = fil.readline()
		ln_number = 1
		try:
			while ln:
				if ln[:4]=='ATOM' or ln[:4]=='HETATM':
					pieces = self.split_fn(ln)
					self.atoms['x'].append( float(pieces[-5]))
					self.atoms['y'].append( float(pieces[-4]))
					self.atoms['z'].append( float(pieces[-3]))
					self.atoms['q'].append( float(pieces[-2]))
					self.atoms['R'].append( float(pieces[-1]))
	
					
				ln = fil.readline()
				ln_number += 1
		
		except:
			pdb.set_trace()

		fil.close()


	def split_fn(self, x):
		pieces = x.split()
		fixed_pieces = []
		for elem in pieces:
			if len(elem)> 10:
				tmp1 = elem[:elem.rfind('-')]
				tmp2 = elem[elem.rfind('-'):]
				fixed_pieces.append(tmp1)
				fixed_pieces.append(tmp2)
			else:
				fixed_pieces.append(elem)

		
		# print(f"line is {x} \n pieces are {fixed_pieces}\n")
		
		return fixed_pieces

class base(load):
	def __init__(self, address, file_name):
		load.__init__(self, address, file_name)

